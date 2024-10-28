import datetime

class TimeTracker:
    def __init__(self, start_time: datetime.time, date: datetime.date):
        """
        Initializes the TimeTracker with a starting datetime.
        
        Args:
            start_time (datetime.datetime): The initial time as a `datetime.datetime` object.
        """
        self.current_time = start_time
        self.current_date = date
        self.current_datetime = datetime.datetime.combine(date, start_time)

    def set_time_per_frame(self, seconds_per_frame: float):
        """
        Sets the amount of time that passes per frame.
        
        Args:
            seconds_per_frame (float): Time in seconds that passes for each frame.
        """
        self.seconds_per_frame = seconds_per_frame

    def increment_time(self, num_frames: int):
        """
        Increments the current time based on the number of frames that passed.
        
        Args:
            num_frames (int): The number of frames that passed.
        """
        # Calculate total seconds to add based on the number of frames and time per frame
        total_seconds = self.seconds_per_frame * num_frames
        # Add the total time to the current datetime
        self.current_datetime += datetime.timedelta(seconds=total_seconds)

    def get_current_time(self) -> str:
        """
        Returns the current time as a string in 'YYYY-MM-DD HH:MM:SS' format.
        
        Returns:
            str: The current time as a formatted string.
        """
        return self.current_time.strftime('%Y-%m-%d %H:%M:%S')
    
    def get_current_datetime(self) -> datetime.datetime: 

        return self.current_datetime
    
    def __str__(self):
        # Customize what gets printed when you print the object
        return f"TimeTracker(current_time={self.get_current_time()})"