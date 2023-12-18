import matplotlib.pyplot as plt

def squares_inside_circle(circle_radius, square_side_length):
    inside_squares = []

    # Iterate through smaller squares
    for i in range(int(-circle_radius / square_side_length), int(circle_radius / square_side_length) + 1):
        for j in range(int(-circle_radius / square_side_length), int(circle_radius / square_side_length) + 1):
            # Coordinates of the current square's corners
            square_x = i * square_side_length
            square_y = j * square_side_length

            # Check if any corner of the square is inside the circle
            corners_inside = 0
            for x_offset in range(2):
                for y_offset in range(2):
                    corner_x = square_x + x_offset * square_side_length
                    corner_y = square_y + y_offset * square_side_length

                    # Calculate distance between the corner and the circle's center
                    distance = (corner_x ** 2 + corner_y ** 2) ** 0.5

                    # Check if the corner is inside the circle
                    if distance < circle_radius:
                        corners_inside += 1
                        break  # No need to check other corners of this square

            # If at least one corner is inside the circle, add the square to the list
            if corners_inside > 0:
                inside_squares.append((square_x, square_y))

    return inside_squares

# Parameters: Circle radius and side length of smaller squares
circle_radius = 5
square_side_length = 0.5  # Change this value as desired

# Calculate smaller squares inside the circle
inside_squares = squares_inside_circle(circle_radius, square_side_length)

# Plotting the circle and identified squares
circle = plt.Circle((0, 0), circle_radius, color='blue', fill=False)
fig, ax = plt.subplots()
ax.add_artist(circle)

for square in inside_squares:
    rect = plt.Rectangle(square, square_side_length, square_side_length, linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

plt.axis('equal')
plt.title('Smaller Squares Inside the Circle (Center at (0,0))')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(visible=True)

#plt.show()

# Save
plt.savefig('images/gpt-bin-test.png')
