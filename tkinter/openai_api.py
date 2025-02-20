from openai import OpenAI
import os
import base64

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

user_image_path = "/Users/yizhengc/dev/Quest_Project_Tennis/tkinter/contact_screenshot1_pose.png"
pro_image_path = "/Users/yizhengc/dev/Quest_Project_Tennis/images/sinner_forehand_contact_pose.png"

user_image = encode_image(user_image_path)
pro_image = encode_image(pro_image_path)

prompt = "Use these images to generate feedback to the user by comparing the arm & body angle and the stance angle with the professional player. The pro is Sinner, and the user is the one in the first image. User Arm & Body Angle: 64, User Stance Angle: 45, Pro Arm & Body Angle: 17, Pro Stance Angle: 59. To clarify, the Arm & Body Angle is the one at the armpit, and the stance angle is the one between the crotch and the 2 knees. Give a maximum 3 sentence overview on how to improve. Don't include the angles and numbers in the feedback, just use simple words."

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{user_image}",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{pro_image}",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0].message.content)