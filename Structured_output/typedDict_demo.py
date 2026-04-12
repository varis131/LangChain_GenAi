from typing import TypedDict
#TypedDict used to create a dictionary with specific keys and value types. It helps in type checking and provides better code readability.  
class user(TypedDict):
    name:str
    age:int

new_user:user={'name':'varis','age':21}    
print(new_user)