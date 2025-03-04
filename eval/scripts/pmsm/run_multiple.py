import eval_dmpe


rpms = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
model_names = ["RLS", "NODE", "PM"]
consider_actions = [False, True]


for consider_action in consider_actions:
    for model_name in model_names:
        for rpm in rpms:
            eval_dmpe.main(rpm, model_name, consider_action)
