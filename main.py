from argparse import ArgumentParser, BooleanOptionalAction

from config_reader import ConfigurationReader

from environment import env_mapping
from policies import policy_mapping
from evaluators import evaluator_mapping


if __name__ == "__main__":
    
    parser = ArgumentParser(description="The Reinforcement Learning Scratch Implementation Project")
    parser.add_argument("--policy-based", dest="policy_based", type=bool, default=False, action=BooleanOptionalAction, help="The algorithm is policy based or not")
    parser.add_argument("--value-based", dest="value_based", type=bool, default=False, action=BooleanOptionalAction, help="The algorithm is value-based or not")
    parser.add_argument("--policy-eval", dest="policy_eval", type=bool, default=False, action=BooleanOptionalAction, help="Only perform policy evaluation")
    
    # "Environment" arguments
    parser.add_argument("--env", dest="env", type=str, default="GridWorld", required=False, help="Environment name")
    
    # Policy arguments
    parser.add_argument("--policy", dest="policy", type=str, default="Random", required=False, help="Policy name")
    
    # Evaluator arguments
    parser.add_argument("--eval", dest="eval", type=str, default="MonteCarlo", required=False, help="Evaluator name")

    args = parser.parse_args()
    
    # ConfigurationReader for reading configuration
    config_reader = ConfigurationReader()
    
    if args.policy_eval:
        
        print(f"Running policy evaluation on {args.env}, using {args.eval} evaluator and {args.policy} policy")
        
        # Requiring an environment, an evaluator, and a policy
        env_config = config_reader.read_environment(args.env)
        eval_config = config_reader.read_evaluator(args.eval)
        policy_config = config_reader.read_policies(args.policy)
        
        env = env_mapping[args.env](**env_config)
        policy = policy_mapping[args.policy](**policy_config)
        eval = evaluator_mapping[args.eval](
            env=env,
            policy=policy,
            **eval_config
        )
        print(eval.eval())
    
    else:
        
        print("Other functionalities are not available.")