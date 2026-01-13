import argparse
import wandb
import pandas as pd

def app_result(config:dict):
    api=wandb.Api()
    run_20=api.run(f"jcoh-research/TR_GNN/")
    run_50=api.run(f"jcoh-research/TR_GNN/")
    run_100=api.run(f"jcoh-research/TR_GNN/")
    run_500=api.run(f"jcoh-research/TR_GNN/")
    run_1000=api.run(f"jcoh-research/TR_GNN/")

    run_batch_20=api.run(f"jcoh-research/TR_GNN/")
    run_batch_50=api.run(f"jcoh-research/TR_GNN/")
    run_batch_100=api.run(f"jcoh-research/TR_GNN/")
    run_batch_500=api.run(f"jcoh-research/TR_GNN/")
    run_batch_1000=api.run(f"jcoh-research/TR_GNN/")

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