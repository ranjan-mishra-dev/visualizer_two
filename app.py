import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# Load the trained model
model = load_model(r'C:\Users\Ranjan Mishra\visualize_folder2\Garbage_Class_new_mo.keras')

# Define waste categories, sections with detailed content, and associated dustbin colors
categories = {
    "Hazardous Waste": {
        'battery': {
            'description': "Batteries are used in a variety of devices to store electrical energy for later use.",
            'segregation': "Store in a dry place, separate from other waste types, and avoid physical damage.",
            'disposal': "Take to a designated battery recycling facility or hazardous waste collection event.",
            'regulations': "Follow local regulations regarding the disposal of hazardous materials like batteries.",
            'type': 'hazardous',
            'color': 'yellow'
        }
    },
    "Glass": {
        'brown-glass': {
            'description': "Brown glass is often used for beverage bottles and provides UV protection for the contents.",
            'segregation': "Separate from other colors of glass and remove any labels or caps before recycling.",
            'disposal': "Take to a glass recycling facility or place in a designated glass recycling bin.",
            'regulations': "Check local recycling guidelines to ensure proper glass segregation.",
            'type': 'recyclable',
            'color': 'green'
        },
        'green-glass': {
            'description': "Green glass is typically used for wine bottles and provides protection against light damage.",
            'segregation': "Separate from other colors of glass and remove any labels or caps before recycling.",
            'disposal': "Take to a glass recycling facility or place in a designated glass recycling bin.",
            'regulations': "Follow local regulations for recycling colored glass.",
            'type': 'recyclable',
            'color': 'green'
        },
        'white-glass': {
            'description': "White glass, or clear glass, is commonly used for food and beverage containers.",
            'segregation': "Separate from colored glass and remove any labels or caps before recycling.",
            'disposal': "Take to a glass recycling facility or place in a designated glass recycling bin.",
            'regulations': "Ensure compliance with local glass recycling guidelines.",
            'type': 'recyclable',
            'color': 'green'
        }
    },
    # Add more categories here...
    "Paper Products": {
        'cardboard': {
            'description': "Cardboard is a thick, sturdy paper product often used for packaging and shipping.",
            'segregation': "Flatten boxes and remove any packing materials before recycling.",
            'disposal': "Place in the paper recycling bin or take to a cardboard recycling facility.",
            'regulations': "Check for local guidelines on recycling cardboard, especially for large quantities.",
            'type': 'recyclable',
            'color': 'green'
        },
        'paper': {
            'description': "Paper is a versatile material used for writing, printing, and packaging.",
            'segregation': "Remove any staples, bindings, or non-paper materials before recycling.",
            'disposal': "Place in the paper recycling bin or take to a paper recycling facility.",
            'regulations': "Ensure compliance with local recycling regulations for paper.",
            'type': 'recyclable',
            'color': 'green'
        }
    },
    "Clothing": {
        'clothes': {
            'description': "Clothing items include wearable garments made from various materials.",
            'segregation': "Sort by material type and condition before donating or recycling.",
            'disposal': "Donate wearable items or take unusable clothing to a textile recycling facility.",
            'regulations': "Follow local guidelines for clothing donations and recycling.",
            'type': 'recyclable',
            'color': 'green'
        },
        'shoes': {
            'description': "Shoes are wearable items made to protect the feet, often made from leather, fabric, or synthetic materials.",
            'segregation': "Pair shoes together and ensure they are clean before donating or recycling.",
            'disposal': "Donate wearable shoes or take unusable shoes to a textile or shoe recycling facility.",
            'regulations': "Check for local donation centers and recycling programs for shoes.",
            'type': 'recyclable',
            'color': 'green'
        }
    },
    "Metals": {
        'metal': {
            'description': "Metal waste includes items made from metals like aluminum, steel, or copper.",
            'segregation': "Separate metals by type if possible and remove any non-metal attachments.",
            'disposal': "Take to a metal recycling facility or place in a designated metal recycling bin.",
            'regulations': "Follow local guidelines for metal recycling and disposal.",
            'type': 'recyclable',
            'color': 'green'
        }
    },
    "Plastics": {
        'plastic': {
            'description': "Plastic waste includes a variety of items made from synthetic materials derived from petrochemicals.",
            'segregation': "Rinse containers and separate by plastic type according to local recycling guidelines.",
            'disposal': "Place in a plastic recycling bin or take to a recycling facility.",
            'regulations': "Check local guidelines for recycling different types of plastics.",
            'type': 'recyclable',
            'color': 'green'
        }
    },
    "General Waste": {
        'trash': {
            'description': "General waste includes non-recyclable materials that are typically sent to landfill.",
            'segregation': "Separate from recyclable materials and hazardous waste.",
            'disposal': "Place in the general waste bin for collection or take to a landfill.",
            'regulations': "Follow local regulations for general waste disposal and reduce waste where possible.",
            'type': 'solid',
            'color': 'orange'
        }
    },
    # Organic Waste Example
    "Organic Waste": {
        'biological': {
            'description': "Biological waste includes food scraps and other organic materials.",
            'segregation': "Separate from other waste types and store in a compost bin.",
            'disposal': "Compost at home or take to a community composting facility.",
            'regulations': "Follow local regulations regarding organic waste disposal.",
            'type': 'organic',
            'color': 'grey'
        }
    }
}

# Flatten the list of categories to use in the model's output
all_categories = [item for section in categories.values() for item in section.keys()]

# Set image dimensions
img_width = 180
img_height = 180

# Streamlit app title
st.title("Waste Segregation Classifier")

# Upload image using Streamlit's file uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image.resize((img_width, img_height))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Determine the predicted waste category
    predicted_category = all_categories[np.argmax(score)]
    
    # Find the section it belongs to and retrieve the detailed content
    for section, items in categories.items():
        if predicted_category in items:
            category_section = section
            category_details = items[predicted_category]
            break

    # Display the results with detailed content
    st.subheader(f'Waste Type: {predicted_category}')
    st.write(f'**Category Section:** {category_section}')
    st.write(f'**Dustbin Color:** {category_details["color"].capitalize()}')
    st.write(f'**Accuracy:** {np.max(score) * 100:.2f}%')

    st.subheader('Description')
    st.write(category_details['description'])

    st.subheader('Segregation Instructions')
    st.write(category_details['segregation'])

    st.subheader('Disposal Methods')
    st.write(category_details['disposal'])

    st.subheader('Regulations or Tips')
    st.write(category_details['regulations'])
