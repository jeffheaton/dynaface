import time


class CalcETC:
    def __init__(self, total_cycles):
        # Initialize the total number of cycles and start time
        self.total_cycles = total_cycles
        self.start_time = time.time()
        self.completed_cycles = 0

    def cycle(self):
        # Check if there are cycles left to complete
        if self.completed_cycles < self.total_cycles:
            # Increment the count of completed cycles
            self.completed_cycles += 1

            # Calculate the elapsed time since the start
            elapsed_time = time.time() - self.start_time

            # Avoid division by zero; calculate average time per cycle
            if self.completed_cycles > 0:
                average_time_per_cycle = elapsed_time / self.completed_cycles
            else:
                average_time_per_cycle = 0

            # Calculate the remaining cycles and ensure it's not negative
            remaining_cycles = max(self.total_cycles - self.completed_cycles, 0)

            # Estimate the remaining time based on the average time per cycle
            estimated_remaining_time = average_time_per_cycle * remaining_cycles
        else:
            # If all cycles are completed, set the remaining time to zero
            estimated_remaining_time = 0

        # Return the formatted time
        return self._format_time(estimated_remaining_time)

    def _format_time(self, seconds):
        # Convert seconds to hours, minutes, and seconds
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)

        # Format the time based on whether hours are present or not
        if hours == 0:
            return f"{minutes:02d}:{seconds:02d}"
        else:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"