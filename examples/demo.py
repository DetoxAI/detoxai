import detoxai
import utils

dl_test, dl_unlearn = utils.get_your_dataloaders()
model = utils.get_your_model()

results = detoxai.debias(
    model,
    dl_unlearn,
    methods=["SAVANIAFT", "RRCLARC"],
    return_type="pareto-front",
)

sail_vis = detoxai.visualization.SSVisualizer(dl_test, model)
sail_vis.visualize_agg(batch_num=0)
sail_vis.show()


scatter_vis = detoxai.visualization.ScatterVisualizer()
scatter_vis.create_plot(results)
scatter_vis.show()
