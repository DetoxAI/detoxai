def require_logger_provided(func):
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "logger") or self.logger is None:
            raise ValueError("A logger must be provided in self.logger")
        return func(self, *args, **kwargs)

    return wrapper


def ensure_metrics_config_not_empty(func):
    def wrapper(self, *args, **kwargs):
        if (
            not hasattr(self, "metrics_config")
            or self.metrics_config is None
            or self.metrics_config == {}
        ):
            raise ValueError("Metrics config has to be initialized and not empty")
        return func(self, *args, **kwargs)

    return wrapper
