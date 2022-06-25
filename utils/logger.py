import datetime
import os


class logger(object):
    def __init__(self, root_path) -> None:
        timenow = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        self.logger_path = os.path.join(root_path, timenow)
        if not os.path.exists(self.logger_path):
            os.makedirs(self.logger_path)
        self.record_path = os.path.join(self.logger_path, "record.log")
        self.config_path = os.path.join(self.logger_path, "config.log")
    
    def printdir(self, dir):
        with open(self.config_path, 'a') as f:
            for k, v in dir.items():
                item = str(k) + ':' + str(v)
                f.write(item)
                f.write('\n')

    def print(self, str):
        with open(self.record_path, 'a') as f:
            f.write(str)
            f.write('\n')
        f.close()
