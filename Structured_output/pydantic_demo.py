from pydantic import BaseModel
from typing import Optional

class student(BaseModel):
    name: str
    age: Optional[int] = None
    grade: Optional[str] = None


new_student={'name':'varis','age': 22, 'grade': 'A'}    
student_obj = student(**new_student)
print(student_obj)
print(type(student_obj))