from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import generate
from torchvision.transforms import Compose, Resize, ToTensor
import torch
from models.encoder import Encoder
from train import init_style_gan

output_size = 512
transform = Compose([
        Resize((1024, 1024)),
        ToTensor(),
    ])

save_image = True
target_dir = "./generated_images/"
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
# model.fc = torch.nn.Linear(model.fc.in_features, output_size)
model = Encoder(3, output_size)
model.load_state_dict(torch.load("./results/checkpoints/encoder_stop.pth"))
model.load_state_dict(torch.load("./results/checkpoints/encoder_200.pth"))
if torch.cuda.is_available():
    model = model.to("cuda")
model.eval()

stylegan = init_style_gan()


def set_img_ui(img):
    img.thumbnail((550, 500))
    img = ImageTk.PhotoImage(img)
    lbl.configure(image=img)
    lbl.image = img

def choose_file():
    filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select image file",
                                          filetypes=(("JPG File", "*.jpg"), ("PNG file", "*.png"), ("All files", "*.")))
    entry1.delete(0, 'end')
    entry1.insert(0, str(filename))
    img = Image.open(filename)
    set_img_ui(img)


root = Tk()
root.title("Image Caption Generator Using Deep Learning")
root.geometry("650x650")

mylabel = Label(root, text=" ", font="24")


def generateImage(mylabel):
    file_name = entry1.get()
    print(file_name)
    img = Image.open(file_name)
    image_tensor = transform(img).unsqueeze(0)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()

    pred_z = model(image_tensor)
    img = stylegan(pred_z, None)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = Image.fromarray(img[0].cpu().numpy(), 'RGB')
    set_img_ui(img)
    if save_image:
        img.save(target_dir + f"gen_image.png")


frm = Frame(root)
frm.pack(side=BOTTOM, padx=10, pady=10)

lbl = Label(root)
lbl.pack()

entry1 = Entry(frm, width=90)

button1 = Button(frm, text="Select Image", command=choose_file, width=20)

button2 = Button(frm, text="Generate Image", command=lambda: generateImage(mylabel), width=20)

entry1.pack()
mylabel.pack()
button1.pack(pady=5)
button2.pack(padx=10, pady=10)

root.mainloop()
