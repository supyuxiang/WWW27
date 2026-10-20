import logging

# 创建一个 logger
logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)   # 设置 logger 的最低级别

# 添加终端 handler（输出到 stdout）
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 添加文件 handler
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 此时 logger.info() 会同时输出到终端和文件
logger.info("这是一条 info 日志")
logger.debug("这是一条 debug 日志")  # 终端看不到（因为 console_handler 级别是 INFO），但文件会记录