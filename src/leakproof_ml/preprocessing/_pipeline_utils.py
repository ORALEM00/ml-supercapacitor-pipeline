from sklearn.pipeline import Pipeline



def _get_pre_model_pipe(pipeline):
    """Slices the pipeline to return all steps except the final 'model'."""
    step_names = list(pipeline.named_steps.keys())
    
    model_idx = step_names.index('model')
    return Pipeline(pipeline.steps[:model_idx])



def _validate_pipeline(pipeline):
    """Ensures the custom pipeline follows the library contract."""
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Custom pipeline must be a sklearn.pipeline.Pipeline object.")
    if 'model' not in pipeline.named_steps:
        raise KeyError("The last step of the pipeline must be named 'model'.")