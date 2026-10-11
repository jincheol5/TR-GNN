import argparse
import torch
from torch.utils.data import DataLoader
from utils import DataUtils,TrainUtils,TemporalGraphDataset,GraphAlgorithm

def main(**kwargs):
    dataset_name=kwargs["dataset_name"]
    source=kwargs["source"]
    batch_size=kwargs["batch_size"]

    graph_df=DataUtils.preprocess_snap_dataset(dataset_name)["graph_df"]
    graph_dataset=TemporalGraphDataset(df=graph_df)
    graph_loader=DataLoader(dataset=graph_dataset,batch_size=batch_size,shuffle=False)

    match kwargs["app_num"]:
        case 1:
            _,_,test_loader=TrainUtils.split_data_loader(graph_loader)
            noisy_batches=DataUtils.load_SNAP_incomplete_noisy_eventstream(dataset_name,batch_size)
            labels=DataUtils.load_SNAP_TR_result(dataset_name,"test",batch_size)["label"][:,source,:].clone()

            # train/val 원본 이력에 noisy test 배치를 누적해 시간순으로 재계산한다.
            history_end=len(graph_dataset)-len(test_loader.dataset)
            history={
                key:values[:history_end]
                for key,values in zip(("src","dst","t"),(graph_dataset.src,graph_dataset.dst,graph_dataset.t))
            }
            empty_TR=torch.zeros(labels.shape[1],dtype=torch.bool)
            empty_t=torch.full((labels.shape[1],),float("inf"))
            mask=torch.ones(labels.shape[1],dtype=torch.bool)
            mask[[0,source]]=False
            batch_accuracies=[]
            for batch_idx,noisy_batch in enumerate(noisy_batches):
                for key in ("src","dst","t"):
                    history[key]=torch.cat((history[key],noisy_batch[key]))
                predicted=GraphAlgorithm.compute_source_TR(
                    source=source,
                    pre_TR=empty_TR,
                    pre_TR_t=empty_t,
                    batch_eventstream=history
                )[mask]
                accuracy=(predicted==labels[batch_idx][mask]).float().mean().item()
                batch_accuracies.append(accuracy)
                print(f"Batch {batch_idx}: accuracy={accuracy:.4%}")
            accuracy=torch.tensor(batch_accuracies,dtype=torch.float64).mean().item()
            print(f"Overall: accuracy={accuracy:.4%}")
            return {"accuracy":accuracy,"batch_accuracies":batch_accuracies}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--dataset_name",
        type=str,
        choices=[
            "CollegeMsg",
            "bitcoin-alpha",
            "bitcoin-otc"
        ],
        default="CollegeMsg"
    )
    parser.add_argument("--source",type=int,default=1)
    parser.add_argument("--batch_size",type=int,default=200)
    args=parser.parse_args()
    app_config={
        "app_num":args.app_num,
        "dataset_name":args.dataset_name,
        "source":args.source,
        "batch_size":args.batch_size
    }
    main(**app_config)
