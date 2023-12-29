
import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from sklearn.cluster import KMeans
import io

# Function to create circle art with 5 shades of gray
def create_circle_art(image, num_shades=5):
    # Convert the image to grayscale and use original dimensions
    grayscale_image = image.convert('L')
    output_size = grayscale_image.size

    # Prepare the output image
    output_image = Image.new('RGB', output_size, (255, 255, 255))
    draw = ImageDraw.Draw(output_image)

    # Cluster pixel intensities for different circle sizes
    pixel_data = np.array(grayscale_image).reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_shades)
    kmeans.fit(pixel_data)
    cluster_centers = sorted(kmeans.cluster_centers_.flatten())

    # Define thresholds and corresponding circle sizes
    thresholds = [int((cluster_centers[i] + cluster_centers[i+1]) / 2) for i in range(len(cluster_centers)-1)]
    circle_sizes = [4, 8, 12, 16, 20]  # Different circle sizes

    # Define grayscale shades
    grayscale_shades = [(40, 40, 40), (80, 80, 80), (120, 120, 120), (160, 160, 160), (200, 200, 200)]

    # Function to map pixel intensity to circle size and color
    def map_pixel_to_circle(pixel_intensity):
        for i, threshold in enumerate(thresholds):
            if pixel_intensity < threshold:
                return circle_sizes[i], grayscale_shades[i]
        return circle_sizes[-1], grayscale_shades[-1]

    # Process each pixel and draw circles
    for x in range(0, output_size[0], max(circle_sizes)):
        for y in range(0, output_size[1], max(circle_sizes)):
            pixel_intensity = grayscale_image.getpixel((x, y))
            circle_size, circle_shade = map_pixel_to_circle(pixel_intensity)
            radius = circle_size // 2

            # Draw the circle
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=circle_shade)

    return output_image

# Streamlit app interface
st.title('Circle Art Generator')

# Image uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Process button
    if st.button('Create Circle Art'):
        # Processing the image to create circle art
        result_image = create_circle_art(image)

        # Display the circle art
        st.image(result_image, caption='Circle Art', use_column_width=True)

        # Save the result image to a bytes buffer
        buf = io.BytesIO()
        result_image.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        # Provide a download button for the processed image
        st.download_button(
            label="Download Image",
            data=byte_im,
            file_name="circle_art.jpg",
            mime="image/jpeg"
        )
