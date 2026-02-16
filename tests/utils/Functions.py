
# Define a function that our model can use.
# ==================================================================================
# Weather:
weather_function_schema = {
    "type": "function",
    "function": {
        "name": "get_current_temperature",
        "description": "Gets the current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. San Francisco",
                },
                "unit": {
                    "type": "string",
                    "description": "Temperature unit, e.g. celsius",
                }
            },
            "required": ["location"],
        },
    }
}

def get_current_temperature(location: str, unit: str = "celsius"):
    """
    Gets the current weather in a given location.

    Args:
        location: The city and state, e.g. "San Francisco, CA" or "Tokyo, JP"
        unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])

    Returns:
        temperature: The current temperature in the given location
        weather: The current weather in the given location
    """
    return {"temperature": 15, "weather": "sunny"}

# ==================================================================================
create_calendar_event_schema = {
            "function": {
                "name": "create_calendar_event",
                "description": "Creates a new calendar event.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "title": {"type": "STRING", "description": "The title of the event."},
                        "datetime": {"type": "STRING", "description": "The date and time in YYYY-MM-DDTHH:MM:SS format."},
                    },
                    "required": ["title", "datetime"],
                },
            }
        }
def create_calendar_event(title: str, datetime: str):
    """Sends an email."""
    print(f"CALENDAR_EVENT function with: title: {title} and datetime: {datetime}")
    return {"response": "Done!"}
# ==================================================================================
send_email_schema = {
        "function": {
            "name": "send_email",
            "description": "Sends an email.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "to": {"type": "STRING", "description": "The recipient email address."},
                    "subject": {"type": "STRING", "description": "The email subject."},
                    "body": {"type": "STRING", "description": "The email body."},
                },
                "required": ["to", "subject"],
            },
        }
    }

def send_email(to: str, subject: str, body: str):
    """Sends an email."""
    print(f"SEND_EMAIL function with to: {to}, subject: {subject}, and body: {body}")
    return {"email sent": "Done!"}
# ==================================================================================

tools_list = [weather_function_schema, create_calendar_event_schema, send_email_schema]
