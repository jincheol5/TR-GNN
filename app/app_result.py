import argparse
import wandb
import pandas as pd

def app_result(config:dict):
    api=wandb.Api()
    ### PR-AUC
    # run_20=api.run(f"jcoh-research/TR-GNN/m0rieti1")
    # run_50=api.run(f"jcoh-research/TR-GNN/h8wqr4a2")
    # run_100=api.run(f"jcoh-research/TR-GNN/jynuh9it")
    # run_500=api.run(f"jcoh-research/TR-GNN/cbk8qbuk")
    # run_1000=api.run(f"jcoh-research/TR-GNN/9mjqpb0j")

    # run_batch_20=api.run(f"jcoh-research/TR-GNN/fof4jjy9")
    # run_batch_50=api.run(f"jcoh-research/TR-GNN/bsfm4s9p")
    # run_batch_100=api.run(f"jcoh-research/TR-GNN/rno2auqw")
    # run_batch_500=api.run(f"jcoh-research/TR-GNN/05yulxi5")
    # run_batch_1000=api.run(f"jcoh-research/TR-GNN/kigkwx07")

    ### ACC
    run_20=api.run(f"jcoh-research/TR-GNN/nggak9g3")
    run_50=api.run(f"jcoh-research/TR-GNN/bhx5hhyj")
    run_100=api.run(f"jcoh-research/TR-GNN/vu2ot5ix")
    run_500=api.run(f"jcoh-research/TR-GNN/zzle3x7s")
    run_1000=api.run(f"jcoh-research/TR-GNN/xese4c9n")

    run_batch_20=api.run(f"jcoh-research/TR-GNN/gsfvksk5")
    run_batch_50=api.run(f"jcoh-research/TR-GNN/rkakl40n")
    run_batch_100=api.run(f"jcoh-research/TR-GNN/khk4efxv")
    run_batch_500=api.run(f"jcoh-research/TR-GNN/0u8zmbfe")
    run_batch_1000=api.run(f"jcoh-research/TR-GNN/g9bw3rxy")

    match config['app_num']:
        case 1:
            """
            App 1.
            result of evaluate_1
            """
            match config['num_nodes']:
                case 20:
                    run=run_20
                case 50:
                    run=run_50
                case 100:
                    run=run_100
                case 500:
                    run=run_500
                case 1000:
                    run=run_1000
            history=run.history(keys=["model","emb","seed","lr","batch_size","acc","macrof1","prauc","mcc"])
            df=pd.DataFrame(history)
            
            # TGN
            df_tgn=df[df["model"]=="tgn"]
            tgn_metric_mean=df_tgn.groupby("emb")[[
                "acc",
                "macrof1",
                "prauc",
                "mcc"
            ]].mean()
            tgn_metric_std=df_tgn.groupby("emb")[[
                "acc",
                "macrof1",
                "prauc",
                "mcc"
            ]].std().round(4)

            # others
            df_others=df[df["model"]!="tgn"]
            others_metric_mean=df_others.groupby("model")[[
                "acc",
                "macrof1",
                "prauc",
                "mcc"
            ]].mean()
            others_metric_std=df_others.groupby("model")[[
                "acc",
                "macrof1",
                "prauc",
                "mcc"
            ]].std().round(4)

            print(f"Evaluate Result:")
            print(tgn_metric_mean)
            print(tgn_metric_std)
            print()
            print(others_metric_mean)
            print(others_metric_std)
            print()

        case 2:
            """
            App 2.
            result of evaluate_2
            """
            match config['num_nodes']:
                case 20:
                    run=run_batch_20
                case 50:
                    run=run_batch_50
                case 100:
                    run=run_batch_100
                case 500:
                    run=run_batch_500
                case 1000:
                    run=run_batch_1000
            history=run.history(keys=["model","seed","lr","batch_size","acc","macrof1","prauc","mcc"])
            df=pd.DataFrame(history)
            
            metric_mean=df.groupby(["model","batch_size"])[[
                "acc",
                "macrof1",
                "prauc",
                "mcc"
            ]].mean()
            print(f"Evaluate Result:")
            print(metric_mean)

if __name__=="__main__":
    """
    Execute app_evaluate
    """
    parser=argparse.ArgumentParser()
    parser.add_argument("--app_num",type=int,default=1)
    parser.add_argument("--num_nodes",type=int,default=20)
    args=parser.parse_args()

    config={
        'app_num':args.app_num,
        'num_nodes':args.num_nodes
    }
    app_result(config=config)