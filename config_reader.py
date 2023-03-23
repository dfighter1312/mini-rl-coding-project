import os
import yaml

from configparser import RawConfigParser


class ConfigurationReader:
    
    base = "base"
    path_to_config = os.path.join(os.path.abspath(os.getcwd()), 'cfg/path_to_config.yml')

    def read_evaluator(self, chosen_eval):
        return self.read_general_configs(
            entity_type="evaluator",
            chosen_entity=chosen_eval,
            entity_type_name="Evaluator"
        )

    def read_environment(self, chosen_env):
        return self.read_general_configs(
            entity_type="environment",
            chosen_entity=chosen_env,
            entity_type_name="Environment"
        )

    def read_agents(self, chosen_agent):
        return self.read_general_configs(
            entity_type="agents",
            chosen_entity=chosen_agent,
            entity_type_name="Agent"
        )
        
    def read_policies(self, chosen_policies):
        return self.read_general_configs(
            entity_type="policies",
            chosen_entity=chosen_policies,
            entity_type_name="Policy"
        )

    def read_general_configs(self, entity_type, chosen_entity, entity_type_name = ""):
        """General configuration retrieval for all functions in the configuration reader.

        Args:
            entity_type (str): Entity type, must be corresponding to sections in the `path_to_config.cfg` file.
            chosen_entity (str): Entity name, must be correspond to item in section in the configuration file.
            entity_type_name (str, optional): For fault logging. Defaults to "".

        Raises:
            KeyError: Raised when the chosen entity is not found.

        Returns:
            Dict: Configurations
        """        
        with open(self.path_to_config, 'r') as stream:
            try:
                path_data = yaml.safe_load(stream)
                entity_type_info = path_data[entity_type]
            except yaml.YAMLError as e:
                raise(e)
    
        try:
            entity_type_path = entity_type_info[self.base] + entity_type_info[chosen_entity]
            abs_path = os.path.join(os.path.abspath(os.getcwd()), entity_type_path)
            with open(abs_path, 'r') as stream:
                entity_data = yaml.safe_load(stream)
            return entity_data if entity_data is not None else {}
        except KeyError as e:
            raise KeyError(f"{entity_type_name} {chosen_entity} does not exist")