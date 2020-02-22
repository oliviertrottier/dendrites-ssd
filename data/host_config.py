import os, socket, json

SCRIPT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
CONFIGS_DIR = os.path.join(SCRIPT_DIR, 'configs/')
HOST_CONFIG_FILENAME = os.path.join(CONFIGS_DIR, 'host_config.json')
HOSTNAME = socket.gethostname()

# Remove random number string that precedes the first '.' in VPN hostname, if any.
if 'vpn' in HOSTNAME:
    Period_positions = HOSTNAME.index('.')
    HOSTNAME = HOSTNAME[Period_positions+1:]

def init_host_config():
    '''
    Initialize host configurations
    :return: None
    '''

    host_config_dict = {HOSTNAME: {'root': ROOT_DIR, 'config': {'dataloader_num_workers': 4}}}

    # Save a default host configuration file.
    if not os.path.exists(HOST_CONFIG_FILENAME):
        with open(HOST_CONFIG_FILENAME, 'w') as fid:
            json.dump(host_config_dict, fid, indent=4)
    else:
        raise Exception('A host configuration file already exists.')


def add_host_config(hostname: str):
    '''
    Add host configuration to the exisiting host configuration file
    :return: Updated host configuration dictionary
    '''
    # Initialize config file
    if not os.path.exists(HOST_CONFIG_FILENAME):
        init_host_config()

    # Load host config and check existence of the input hostname
    with open(HOST_CONFIG_FILENAME) as fid:
        host_configs = json.load(fid)
    if hostname in host_configs:
        raise Exception('The configuration already exists for {}. Edit the JSON file instead.'.format(hostname))

    # Add a new default configuration and save
    new_host_config = {hostname: {'root': ROOT_DIR, 'config': {'dataloader_num_workers': 4}}}
    host_configs.update(new_host_config)
    with open(HOST_CONFIG_FILENAME, 'w') as fid:
        json.dump(host_configs, fid, indent=4)

    return new_host_config[hostname]


def get_host_config():
    '''
    Read the host configuration file
    :return: None
    '''

    with open(HOST_CONFIG_FILENAME) as fid:
        host_configs = json.load(fid)

    if HOSTNAME in host_configs:
        return host_configs[HOSTNAME]
    else:
        # If HOSTNAME doesn't exist in the local configuration file, add it to the list.
        return add_host_config(HOSTNAME)