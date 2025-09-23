#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: leicheng
"""
import glob
import datetime

def extract_start_time(path):
    csv_files = glob.glob(path + "/*.csv")
    for csv_file in csv_files:
        # Read the CSV file
        with open(csv_file, 'r') as file:
            lines = file.readlines()

        # Find the line containing "Capture start time"
        start_time_line = [line for line in lines if 'Capture start time' in line][0]

        # Extract the time string
        start_time_str = start_time_line.split('-')[-1].strip()

        # Parse the time string into a datetime object
        start_time = datetime.datetime.strptime(start_time_str, '%a %b %d %H:%M:%S %Y')

        # Convert the time to UNIX EPOCH time
        epoch_time = start_time.timestamp()

        # Print the results
        print("Capture start time: ", start_time)
        print("UNIX EPOCH time: ", epoch_time)
    
    return epoch_time



if __name__ == "__main__":
    #path = "/xdisk/caos/leicheng/my_rawdata_0624/0624/lei-leicar/"
    path = "/xdisk/caos/leicheng/my_rawdata_0624/0624/jiahao-hao"
    epoch_time = extract_start_time(path)

