

def select_best_model(model_results,metric = "val_psnr"):
    
    best_model_name = None
    best_checkpoint = None
    best_metrics = None

    best_score = -float("inf")

    for model_name,result in model_results.items():

        metrics = result["best_metrics"]

        score = metrics[metric]

        if score > best_score:
            best_score = score

            best_model_name = model_name

            best_checkpoint = result["checkpoint"]

            best_metrics = metrics

    return(
        best_model_name,
        best_checkpoint,
        best_metrics
        )


def rank_models(
        model_results,
        metric = "val_psnr",
        higher_is_better = True
):

    rank_models = sorted(
        model_results.items(),
        key = lambda x: x[1]["best_metrics"][metric],
        reverse = higher_is_better
    )
    
    return rank_models