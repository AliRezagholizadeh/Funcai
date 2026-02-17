from enum import Enum

class CHATSIDE(Enum):
    User = "user"
    Agent = "agent"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value
    
class Role(Enum):
    User: str = "user"
    Assistant: str = "assistant"
    Developer: str = "developer"

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


if __name__ == "__main__":
    print(Role.User)