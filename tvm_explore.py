
## Author Manoj Kumar

from ast import arg
from enum import auto
from inspect import trace
from numpy import dtype
import torch
import torchvision
import torchvision.transforms as transforms
import os
import time
import argparse
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import logging
#logging.getLogger('autotvm').setLevel(logging.DEBUG)
#logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

assert(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Cifa10DataSet():
    def __init__(self, testbatch, trainbatch=64) -> None:
        self.width = 224
        self.height = 224
        self.outfeatures = 10

        # Loads CIFAR10 dataset.
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  

        self.my_preprocess2 = transforms.Compose(
            [
                transforms.Resize(self.height),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.sample_to_take = 10000

        if testbatch > 50:
            self.sample_to_take = 2000


        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=self.my_preprocess2)
        
        #self.trainsubset = torch.utils.data.Subset(self.trainset, range(0, 10000))
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=trainbatch,
                                                shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=self.my_preprocess2)
        #self.testsubset = torch.utils.data.Subset(self.testset, range(0,self.sample_to_take))
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=testbatch,
                                                shuffle=False, num_workers=2)

class AnimalClassification():
    def __init__(self, testbatch, img_shape, trainbatch=16) -> None:
        self.width = img_shape
        self.height = img_shape
        self.outfeatures = 3
        
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  
        
        self.my_preprocess = transforms.Compose(
            [
                transforms.Resize(self.height),
                transforms.CenterCrop(self.height),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.my_preprocess2 = transforms.Compose(
            [
                transforms.Resize(self.height),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.traindata = torchvision.datasets.ImageFolder(root='Dataset/training', transform=self.my_preprocess2)
        self.trainloader = torch.utils.data.DataLoader(self.traindata, batch_size=trainbatch,
                                                shuffle=True, num_workers=4)

        self.testdata = torchvision.datasets.ImageFolder(root='Dataset/validation', transform=self.my_preprocess2)
        self.testloader = torch.utils.data.DataLoader(self.testdata, batch_size=testbatch,
                                                shuffle=True, num_workers=4)


class MetaData():
    def __init__(self, test_batch, model_name, tvm_enable, autotune_tvm, img_shape) -> None:
        self.dataset = AnimalClassification(test_batch, img_shape)

        # Number of distinct number labels, [0..9]
        self.NUM_CLASSES = 10
        self.samples = 100000
        self.seed = 10
        self.width = self.dataset.width
        self.height = self.dataset.height
        self.tvm_enable = tvm_enable
        self.autotune_tvm = autotune_tvm
        self.out_classes = self.dataset.outfeatures

        # Number of examples in each training batch (step)
        self.TEST_BATCH = test_batch
        self.epochs = 3
        self.test_iteration = 1
        self.use_cpu = False

        # Number of training steps to run
        self.TRAIN_STEPS = 1000
        self.trainloader = self.dataset.trainloader
        self.testloader = self.dataset.testloader

        self.model_name = model_name
        self.model_path = f'saved_model/cifar_net_{self.model_name}.pt'

        if self.tvm_enable:
            self.tvm_status = 'enabled'
        else:
            self.tvm_status = 'disabled'

class ModelPool():
    def __init__(self, params) -> None:
        self.params = params 
    
    def transfer_learning(self, model):
        for layer in model.parameters():
            layer.requires_grad = False
        
        model.classifier[6].out_features = self.params.out_classes
        return model.to(device)

    def create_vgg16(self):
        model = torchvision.models.vgg16(pretrained=True)
        for layer in model.parameters():
            layer.requires_grad = False
        
        model.classifier[6].out_features = self.params.out_classes
        return model.to(device)

    def create_resnet50_bk(self):
        model = torchvision.models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        for layer in model.parameters():
            layer.requires_grad = False
        
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.params.out_classes)
        )
        #print(model)

        return model.to(device)
    
    def create_resnet50(self):
        model = torchvision.models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        for layer in model.parameters():
            layer.requires_grad = False
        
        model.fc = nn.Linear(in_features=2048, out_features=self.params.out_classes, bias=True)
        print(model)

        return model.to(device)

    def create_resnet101(self):        
        model = torchvision.models.resnet101(pretrained=True)
        for layer in model.parameters():
            layer.requires_grad = False
        
        model.fc.out_features = self.params.out_classes
        return model


    def generate_model(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 5 * 5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, self.params.out_classes)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = torch.flatten(x, 1)  # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        model = Net().to(device)
        return model

class PlayGround():
    def __init__(self, params:MetaData) -> None:
        self.params = params
    
    def get_autotune_network(self, name, batch_size, layout="NHWC", dtype="float32"):
        if layout == "NHWC":
            image_shape = (self.params.width, self.params.height, 3)
        elif layout == "NCHW":
            image_shape = (3, self.params.width, self.params.height)
        else:
            raise ValueError("Invalid layout: " + layout)
        
        output_shape = (batch_size, 1000)
        input_shape = (batch_size,) + image_shape
        
        if name.startswith("resnet-"):
            n_layer = int(name.split("-")[1])
            mod, params = relay.testing.resnet.get_workload(
                num_layers=n_layer,
                batch_size=batch_size,
                layout=layout,
                dtype=dtype,
                image_shape=image_shape,
            )
        else:
            raise ValueError("autotune availbale only for resnet")
        
        return mod, params, input_shape, output_shape
        
        

    def train(self, model):
        global model_name
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for epoch in tqdm(range(self.params.epochs)):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.params.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                score, preds = torch.max(outputs, 1)
                accuracy = (preds == labels).sum().item() / inputs.shape[0]

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item() / inputs.shape[0]
                if ((i % 20 == 0) and (i != 0)):    # print every 2000 mini-batches
                    print(f'{epoch} loss: {running_loss} : Accuracy = {accuracy}')
                    running_loss = 0.0
                    
                    
        input_shape = [self.params.TEST_BATCH, 3, self.params.width, self.params.height]
        input_data = torch.randn(input_shape).to(device)
        scripted_model = torch.jit.trace(model, input_data)
        #scripted_model = torch.jit.script(model)
        scripted_model.save(self.params.model_path)
        return scripted_model
    
    def warmup(self, model):
        random_gen_img = torch.rand(self.params.TEST_BATCH, 3, self.params.width, self.params.height)
        random_gen_img =  random_gen_img.to(device)
        warmup_itr = 5
        for _ in range(warmup_itr):
            model(random_gen_img)

        return model

    def tvm_warmup(self, model):
        random_gen_img = torch.rand(self.params.TEST_BATCH, 3, self.params.width, self.params.height)
        #random_gen_img =  random_gen_img.to(device)
        warmup_itr = 5
        for _ in range(warmup_itr):
            model.set_input("input1", tvm.nd.array(random_gen_img.numpy()))
            model.run()

        return model
    
    def inference(self, model):
        inference_time = []
        accuracy_list = []
        confidence = []
        results = pd.DataFrame()

        for iteration in range(self.params.test_iteration):
            accuracy_list.clear()
            inference_time.clear()
            confidence.clear()
            
            for i, data in tqdm(enumerate(self.params.testloader)):
                input, labels = data 
                print("shape = ", input.shape)
                inputs, labels = data[0].to(device), data[1].to(device)
                start = time.time()
                out = model(inputs)
                end = time.time()
                confidence.append(torch.nn.functional.softmax(out, dim=1).cpu().detach().numpy().max(axis=1))
                score, preds = torch.max(out, 1)
                acc = (preds == labels).sum().item()/ inputs.shape[0]
                accuracy_list.append(acc)
                inference_time.append((end-start)/inputs.shape[0])
            
            results[str(iteration) + "_time"] = inference_time
            results[str(iteration) + "_acc"] = accuracy_list
            results[str(iteration) + "_conf"] = confidence

            
        dataframe_name = f'TVM_{self.params.tvm_status}_{self.params.model_name}_bch{str(self.params.TEST_BATCH)}.csv'

        results.to_csv(dataframe_name)
        print("Data is saved in ", dataframe_name)
        self.get_stats(dataframe_name)
    
    def autotune(self, mod, params, target):
        log_file = "%s-B%d-%s.json" % (self.params.model_name, self.params.TEST_BATCH, target.kind.name)
        image_shape = (self.params.width, self.params.height, 3)
        output_shape = (self.params.TEST_BATCH, 10)
        input_shape = (self.params.TEST_BATCH,) + image_shape

        tasks, task_weights = tvm.auto_scheduler.extract_tasks(mod["main"], params, target)
        for idx, task in enumerate(tasks):
            print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
            print(task.compute_dag)
        
        print("Begin tuning...")
        measure_ctx = tvm.auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

        tuner = tvm.auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = tvm.auto_scheduler.TuningOptions(
            num_measure_trials=200,  # change this to 20000 to achieve the best performance
            runner=measure_ctx.runner,
            measure_callbacks=[tvm.auto_scheduler.RecordToFile(log_file)],
        )

        tuner.tune(tune_option)
        print("Compile...")
        with tvm.auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(mod, target=target, params=params)
        
        return lib

    def evaluate(self, model):
        print("#### Evaluation Started ####")
        if self.params.tvm_enable:
            model.eval()

            input_name = "input1"
            shape_list = [(input_name, (self.params.TEST_BATCH,3,self.params.height,self.params.width))]
            md, model_params = relay.frontend.from_pytorch(model, shape_list)
            if self.params.use_cpu:
                print("DEVICE : LLVM")
                target = tvm.target.Target("llvm", host="llvm")
                dev = tvm.cpu(0)
            else:
                print("DEVICE : CUDA")
                target = tvm.target.cuda(arch='sm_61')
                dev = tvm.cuda(0)

            #dev = tvm.device(str(target))
            if self.params.autotune_tvm:
                lib = self.autotune(md, model_params, target)
            else:
                with tvm.transform.PassContext(opt_level=3):
                    lib = relay.build(md, target=target, params=model_params)
            
            model_graph = graph_executor.GraphModule(lib["default"](dev))

            model_graph = self.tvm_warmup(model_graph)

            inference_time = []
            accuracy_list = []
            confidence = []

            for i, data in tqdm(enumerate(self.params.testloader)):
                input, labels = data 
                print("shape = ", input.shape)
                if input.shape != (self.params.TEST_BATCH,3, self.params.height,self.params.width):
                    print(f"Last batch of size {input.shape} ignored")
                    break

                dtype = "float32"
                model_graph.set_input(input_name, tvm.nd.array(input.numpy().astype(dtype)))
                start_t = time.time()
                model_graph.run()
                end_t = time.time()
                # Get outputs
                tvm_output = model_graph.get_output(0)
                preds = np.argmax(tvm_output.numpy(), axis=1)
                confidence.append(torch.nn.functional.softmax(torch.from_numpy(tvm_output.numpy()), dim=1).numpy().max(axis=1))
                acc = (preds == labels.numpy()).sum().item()/ input.shape[0]
                accuracy_list.append(acc)
                inference_time.append((end_t - start_t)/input.shape[0])
            
            results = pd.DataFrame()
            results["0_time"] = inference_time
            results["0_acc"] = accuracy_list
            results["Itr0_conf"] = confidence
            dataframe_name = f'TVM_{self.params.tvm_status}_{self.params.model_name}_bch{str(self.params.TEST_BATCH)}.csv'
            results.to_csv(dataframe_name)
            print("Data is saved in ", dataframe_name)
            self.get_stats(dataframe_name)

        else:
            model.eval()
            model = self.warmup(model)
            self.inference(model)


    def get_stats(self, csv_file):
        trace_d = pd.read_csv(csv_file)
        trace_d.drop(axis=1, inplace=True, index=0)
        time_columns = [col for col in trace_d.columns if 'time' in col]
        mean_time = trace_d[time_columns].mean().mean()
        print(f"Inference took {mean_time * 1000}ms : accuracy mean {trace_d['0_acc'].mean()}")


if __name__ == "__main__":
    # python tvm_explore.py --create-model resnet50 --infe-batch 64
    # python tvm_explore.py --create-model resnet50 --infe-batch 64 --tvm
    parser = argparse.ArgumentParser()
    parser.add_argument('--tvm', action='store_true',
                        default=False, help="Set true to enable jit_compiler")
    parser.add_argument('--infe-batch', default=1,
                        help="Set the batch size used in inference", type=int)
    parser.add_argument('--image-shape', default=224,
                        help="Set image share for resize", type=int)
    parser.add_argument('--only-train', action='store_true',
                        default=False, help="Train to save model")
    parser.add_argument('--create-model', default='resnet50',
                        help='set which model to use for inference')
    parser.add_argument('--autotune-tvm', action='store_true', default=False,
                        help='set true to enable autotune')
    args = parser.parse_args()

    metadata = MetaData(model_name=args.create_model, test_batch=args.infe_batch, tvm_enable=args.tvm, autotune_tvm=args.autotune_tvm, img_shape=args.image_shape)
    modelpool = ModelPool(metadata)

    model_fn = {'resnet50': modelpool.create_resnet50,
                'resnet101': modelpool.create_resnet101,
                'vgg16': modelpool.create_vgg16,
                'ownmodel': modelpool.generate_model
                }

    print("Using Moldel ", metadata.model_name)    
    ground = PlayGround(metadata)

    if args.only_train:
        model = model_fn[args.create_model]()
        ground.train(model)
    else:
        model = torch.jit.load(metadata.model_path)        
        ground.evaluate(model)