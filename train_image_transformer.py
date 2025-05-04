from datasets import load_dataset
import csv
from datasets import load_dataset
import json
from PIL import Image, ImageDraw


getsum = load_dataset('json', data_files={'train': 'letsee.json'})
annotations = getsum['train'][1]['objects']

print(getsum['train'][0])

categories = getsum['train'][1]['objects']['category']
print(categories)

id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v:k for k,v in id2label.items()}


'''
#for this to work the bbox has to be in a single list ratherthan a list of list which is needed later 
for i in range(1,2):
    print(i)
    box = annotations['bbox']
    print(box)
    image = Image.open(getsum["train"][1]['image'])
    draw = ImageDraw.Draw(image)
    class_idx = categories
    x,y,w,h = tuple(box)
    draw.rectangle((x,y,x+w,y+h),outline='red', width=1)
    draw.text((x,y),'powerline', fill='white')
    #image.save(str(i)+'.jpg')
    image.show()
'''

from transformers import AutoImageProcessor

checkpoint = "facebook/detr-resnet-50"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

import albumentations
import numpy as np
import torch

transform = albumentations.Compose(
    [
        albumentations.Resize(480, 480),
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)

def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations

# transforming a batch
def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["objects"]):
    	with Image.open(image) as im:

        	image = np.array(im.convert("RGB"))[:, :, ::-1]
        	out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])

        	area.append(objects["area"])
        	images.append(out["image"])
        	bboxes.append(out["bboxes"])
        	categories.append(out["category"])

    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]

    return image_processor(images=images, annotations=targets, return_tensors="pt")


#transform_aug_ann(getsum['train'])

getsum['train'] = getsum['train'].with_transform(transform_aug_ann)

#print(getsum['train'][0])



def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch

from transformers import AutoModelForObjectDetection

model = AutoModelForObjectDetection.from_pretrained(
    checkpoint,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="detr-250_300e_plus_two_flip",
    per_device_train_batch_size=5,
    num_train_epochs=50,
    fp16=True,
    save_steps=200,
    logging_steps=50,
    #would not let me do below -5, -4
    #learning_rate=1e-4,
   # weight_decay=1e-3,   
    #learning_rate=1e-5,
    #weight_decay=1e-4,
    learning_rate=1e-6,
    weight_decay=1e-7,
    #learning_rate=1e-9,
    #weight_decay=1e-8,   
    #learning_rate=1e-7,
    #weight_decay=1e-4,  
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=getsum["train"],
    tokenizer=image_processor,
)


print("detr-250_300e_plus_two_flip")
trainer.train()

