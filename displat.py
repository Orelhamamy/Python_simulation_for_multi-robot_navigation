#!/usr/bin/env python3

import pygame
import random

# Initialize Pygame
pygame.init()

# Set the width and height of the screen
width = 800
height = 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Range Sensor Simulation")

# Set the colors
black = (0, 0, 0)
white = (255, 255, 255)

# Set the initial position and range of the sensor
sensor_x = width // 2
sensor_y = height // 2
sensor_range = 100

# Main program loop
running = True
clock = pygame.time.Clock()

while running:
    # Limit the frame rate
    clock.tick(10)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill(white)

    # Simulate range sensor data
    range_data = random.randint(0, sensor_range)

    # Draw the range sensor
    pygame.draw.circle(screen, black, (sensor_x, sensor_y), sensor_range, 1)
    pygame.draw.circle(screen, black, (sensor_x, sensor_y), range_data)

    # Update the screen
    pygame.display.flip()

# Quit the program
pygame.quit()

