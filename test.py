from chat_app import ChatApp
from datetime import datetime

instance = ChatApp()

query1 = "Where have you been?"
query2 = "Where do you live?"
query3 = "It's a trap"

print(datetime.now())
print(instance.get_response(query1))
print(datetime.now())
print(instance.get_response(query2))
print(datetime.now())
print(instance.get_response(query3))
print(datetime.now())
