def write_to_file(filename, data, mode='a'):
    """
    Writes or appends data to a text file in a single line.
    
    Args:
    filename (str): The name or path of the file.
    data (list): The data to write to the file (e.g., [frame_iterator, class_pairs]).
    mode (str): 'w' to overwrite the file, 'a' to append to the file. Default is 'a'.
    """
    # Open the file in the selected mode ('w' for overwrite, 'a' for append)
    with open(filename, mode) as file:
        # Join the data elements as strings and write them in a single line
        line = ' '.join([str(item) for item in data])
        file.write(line + '\n')  # Write in the same line with a newline at the end

def read_file_by_lines(file_name, num_lines, start_line=0):
    """
    Reads the specified number of lines from a file, starting from a given line.
    
    Args:
    file_name (str): The name or path of the file to read.
    num_lines (int): The number of lines to read from the file.
    start_line (int): The line number to start reading from (0-based).
    
    Returns:
    list: A list of the lines read from the file.
    """
    lines = []
    read_lines = True
    with open(file_name, 'r') as file:
        # Skip lines until reaching the start_line
        for _ in range(start_line):
            file.readline()

        # Now read the specified number of lines
        for _ in range(num_lines):
            line = file.readline()
            if not line:
                read_lines = False 
                break  # Stop if there are no more lines to read
            lines.append(line.strip())  # Add the line (without newline characters)
    
    return lines, read_lines




