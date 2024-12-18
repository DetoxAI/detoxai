import lightning as L
import numpy as np


from ..model_correction import ModelCorrectionMethod
from src.models.SklearnWrapper import SklearnWrapper
from src.core.collators.base_collator import BaseCollator


class ThresholdOptimizer(ModelCorrectionMethod):
    def __init__(
        self,
        model: L.LightningModule,
        experiment_name: str,
        device: str,
        base_collator: BaseCollator,
        logger,
    ):
        super().__init__(model.model, experiment_name, device)
        self.scikit_model = SklearnWrapper(model, collator=base_collator, logger=logger)

    def apply_model_correction(
        self,
        constraints: str,
        objective: str,
        y_unlearn: np.ndarray,
        contains_protected_attrubutes_unlearn: np.ndarray,
        X_unlearn: np.ndarray,
    ) -> None:
        self.scikit_model.virtual_fit()
        self.postprocess_est = ThresholdOptimizer(
            estimator=self.scikit_model,
            constraints=constraints,
            objective=objective,
            prefit=True,
            predict_method="predict",
        )

        # Ensure sensitive features have more than one unique label
        unique_labels = np.unique(contains_protected_attrubutes_unlearn)
        if len(unique_labels) > 1:
            # Check if each unique label in sensitive features has more than one unique target label
            valid_sensitive_feature = all(
                len(
                    np.unique(y_unlearn[contains_protected_attrubutes_unlearn == label])
                )
                > 1
                for label in unique_labels
            )
            if valid_sensitive_feature:
                self.postprocess_est.fit(
                    X_unlearn,
                    y_unlearn,
                    sensitive_features=contains_protected_attrubutes_unlearn,
                )
            else:
                print(
                    "Each unique label in sensitive features must have more than one unique target label."
                )
        else:
            print("Sensitive feature values must have more than one unique label.")

    def get_corrected_model(self) -> SklearnWrapper:
        return self.scikit_model


# from threshold_optimizer import ThresholdOptimizer
# from src.core.collators.base_collator import BaseCollator
# from src.models.SklearnWrapper import SklearnWrapper
# import lightning as L
# import numpy as np

# # Initialize components
# model = L.LightningModule
# base_collator = BaseCollator()
# logger = ...  # Initialize your logger

# # Wrap the model
# sk_model = SklearnWrapper(model, collator=base_collator, logger=logger)
# sk_model.virtual_fit()

# # Initialize ThresholdOptimizer
# threshold_optimizer = ThresholdOptimizer(
#     model=model,
#     experiment_name="threshold_optimization_experiment",
#     device="cuda",
#     base_collator=base_collator,
#     logger=logger
# )

# # Apply model correction
# threshold_optimizer.apply_model_correction(
#     constraints="equalized_odds",
#     objective="accuracy_score",
#     y_unlearn=y_dat_unlearn,
#     contains_protected_attrubutes_unlearn=pas_dat_unlearn,
#     X_unlearn=X_dat_unlearn
# )

# # Retrieve the corrected model
# corrected_model = threshold_optimizer.get_corrected_model()
