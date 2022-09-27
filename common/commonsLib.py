import logging
import os


class LogItem:

    def __init__(self, Message, Level, ObjectType, ObjectData):
        self.Message = Message
        self.Level = Level
        self.ObjectType = ObjectType
        self.ObjectData = ObjectData


class NoTrashFilter(logging.Filter):
    def __init__(self, loggingLevel):
        super().__init__()
        self.noTrashArray = ["_log", "LogInput", "LogResult", "Information", "Error", "Debug", "log_exception"]
        self.loggingLevel = loggingLevel

    def filter(self, record):
        return record.funcName in self.noTrashArray or record.levelno >= self.loggingLevel


class loggerFileAux:
    def __init__(self, debug_mode):
        self.DEBUG_MODE = debug_mode
        self.LOG_LIST = []

    def Log(self, level, message):
        if self.DEBUG_MODE:
            self.LOG_LIST.append({"level": level, "message": message})


class loggerElk:
    def __init__(self, owner__name__):

        switcher = {
            "CRITICAL": 50,
            "ERROR": 40,
            "WARNING": 30,
            "INFO": 20,
            "DEBUG": 10
        }
        self.serviceName = str(owner__name__)

        enableKibana = self.__get_boolean_os_var("ELK_ENABLED")
        enableFile = self.__get_boolean_os_var("FILE_ENABLED")

        try:
            self.lib_lob_level = str(os.environ["LIBRARIES_LOG_LEVEL"])
        except:
            print('ERROR GETTING THE ENV_VAR LIBRARIES_LOG_LEVEL... \'ERROR\' BY DEFAULT')
            self.lib_lob_level = "ERROR"
        try:
            logLevel = os.environ["LOG_LEVEL"]
        except:
            print('ERROR GETTING THE ENV_VAR LOG_LEVEL... \'DEBUG\' BY DEFAULT')
            logLevel = "DEBUG"

        logging.basicConfig(filemode='a')
        logging.getLogger().setLevel(logging.FATAL)
        self.logger = logging.getLogger()
        self.logger.handlers = []
        self.logger.setLevel(logLevel)
        self.logger.addFilter(NoTrashFilter(switcher.get(self.lib_lob_level, "")))
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s (%(process)s %(threadName)s) - %(funcName)s -> %(lineno)s - %(message)s')

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.addFilter(NoTrashFilter(switcher.get(self.lib_lob_level, "")))
        ch.setLevel(logLevel)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        # create file handler which logs even debug messages
        if enableFile:
            try:
                logFile = os.environ["LOG_FILE"]
                fh = logging.FileHandler(logFile)
                fh.addFilter(NoTrashFilter(switcher.get(self.lib_lob_level, "")))
                fh.setLevel(logLevel)
                fh.setFormatter(formatter)
                self.logger.addHandler(fh)
            except Exception as e:
                self.logger.warning("LOG_FILE env-var not provided or can't write the file::{}.".format(e))
                print("WARNING!: LOG_FILE env-var not provided or can't write the file::{}.".format(e))

        else:
            self.elkEnabled = False

    def LogResult(self, message, ObjectData, extraAttrs=None):
        li = LogItem(message, 'Information', "result", ObjectData)
        self.logger.info(message + " - result - " + str(ObjectData))

    def LogInput(self, message, ObjectData, extraAttrs=None):
        li = LogItem(message, 'Information', "input", ObjectData)
        self.logger.info(message + " - input - " + str(ObjectData))

    def Information(self, message, extraAttrs=None):
        li = LogItem(message, 'Information', "trace", "")
        self.logger.info(message)

    def Debug(self, message, extraAttrs=None):
        li = LogItem(message, 'Debug', "trace", "")
        self.logger.debug(message)

    def Error(self, message, sysExecInfo=None):
        error = list()
        if sysExecInfo is not None:
            for e in sysExecInfo:
                if hasattr(e, 'tb_frame'):
                    error.append(str(e.tb_frame))
                else:
                    error.append(str(e))
        li = LogItem(message, 'Error', "trace", error)
        error.insert(0, message)

        self.logger.exception(str(error))

    @staticmethod
    def __get_boolean_os_var(os_var):
        if not os_var in os.environ:
            return False
        nat_var = os.environ[os_var]
        if isinstance(nat_var, str):
            return True if nat_var == "True" else False
        else:
            return nat_var
