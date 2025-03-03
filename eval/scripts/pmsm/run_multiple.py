import eval_dmpe


rpms = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
model_names = ["RLS", "NODE", "PM"]
consider_actions = [False, True]

for rpm in rpms:
    for model_name in model_names:
        for consider_action in consider_actions:
            eval_dmpe.main(rpm, model_name, consider_action)
