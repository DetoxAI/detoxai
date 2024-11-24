# Wrapper for requiring activations and CAVs to be computed before applying model correction
def require_activations_and_cav(func):
    def wrapped(self, cav_layer: str, *args, **kwargs):
        if not hasattr(self, "activations"):
            raise ValueError(
                "Activations must be computed before applying model correction"
            )

        if not hasattr(self, "cav"):
            raise ValueError("CAVs must be computed before applying model correction")

        return func(self, cav_layer, *args, **kwargs)

    return wrapped
