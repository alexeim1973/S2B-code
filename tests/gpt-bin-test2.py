import matplotlib.pyplot as plt
import numpy as np

def squares_inside_circle(circle_radius, square_side_length):
    all_squares = []
    square_colors = []

    # Calculate the maximum range based on the circle's radius
    max_range = int(2 * circle_radius / square_side_length)

    # Generate colors for all squares inside the big square
    for i in range(-max_range, max_range):
        for j in range(-max_range, max_range):
            # Assign color values based on the condition
            if -square_side_length * 2 <= j * square_side_length <= -square_side_length or square_side_length <= j * square_side_length <= square_side_length * 2:
                square_colors.append(50)
            else:
                square_colors.append(5)
            all_squares.append((i * square_side_length / 2, j * square_side_length / 2))

    # Identify squares inside the circle
    inside_squares = [square for square in all_squares
                      if (square[0] ** 2 + square[1] ** 2) ** 0.5 < circle_radius]

    # Filter corresponding colors for squares inside the circle
    inside_square_colors = [color for square, color in zip(all_squares, square_colors)
                            if (square[0] ** 2 + square[1] ** 2) ** 0.5 < circle_radius]

    return inside_squares, inside_square_colors

# Parameters: Circle radius and side length of smaller squares
circle_radius = 5
square_side_length = 0.5  # Change this value as desired

# Calculate smaller squares inside the circle and their corresponding colors
inside_squares, square_colors = squares_inside_circle(circle_radius, square_side_length)

# Plotting the circle and identified squares with colors
circle = plt.Circle((0, 0), circle_radius, color='blue', fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle)

for square, color in zip(inside_squares, square_colors):
    rect = plt.Rectangle(square, square_side_length, square_side_length, linewidth=1, edgecolor='black',
                         facecolor=plt.cm.magma(color / 100))  # Using magma colormap
    ax.add_patch(rect)

plt.axis('equal')
plt.title('Smaller Squares Inside the Circle (Center at (0,0)) with Color Map')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(visible=True)
plt.show()

# Save
plt.savefig('images/gpt-bin-test2.png')