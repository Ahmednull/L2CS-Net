import os, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from l2cs import select_device, natural_keys, gazeto3d, angular, getArch, L2CS, Gaze360, Mpiigaze


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze estimation using L2CSNet .')
     # Gaze360
    parser.add_argument(
        '--gaze360image_dir', dest='gaze360image_dir', help='Directory path for gaze images.',
        default='datasets/Gaze360/Image', type=str)
    parser.add_argument(
        '--gaze360label_dir', dest='gaze360label_dir', help='Directory path for gaze labels.',
        default='datasets/Gaze360/Label/test.label', type=str)
    # mpiigaze
    parser.add_argument(
        '--gazeMpiimage_dir', dest='gazeMpiimage_dir', help='Directory path for gaze images.',
        default='datasets/MPIIFaceGaze/Image', type=str)
    parser.add_argument(
        '--gazeMpiilabel_dir', dest='gazeMpiilabel_dir', help='Directory path for gaze labels.',
        default='datasets/MPIIFaceGaze/Label', type=str)
    # Important args -------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--dataset', dest='dataset', help='gaze360, mpiigaze',
        default= "gaze360", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path to the folder contains models.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4-lr', type=str)
    parser.add_argument(
        '--evalpath', dest='evalpath', help='path for the output evaluating gaze test.',
        default="evaluation/L2CS-gaze360-_loader-180-4-lr", type=str)
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=100, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, [ResNet50], ''ResNet101, ResNet152, Squeezenet_1_0, Squeezenet_1_1, MobileNetV2',
        default='ResNet50', type=str)
    # ---------------------------------------------------------------------------------------------------------------------
    # Important args ------------------------------------------------------------------------------------------------------
    args = parser.parse_args()
    return args


def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    gpu = select_device(args.gpu_id, batch_size=args.batch_size)
    batch_size=args.batch_size
    arch=args.arch
    data_set=args.dataset
    evalpath =args.evalpath
    snapshot_path = args.snapshot
    bins=args.bins
    angle=args.angle
    bin_width=args.bin_width

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


    
    if data_set=="gaze360":
        
        gaze_dataset=Gaze360(args.gaze360label_dir,args.gaze360image_dir, transformations, 180, 4, train=False)
        test_loader = torch.utils.data.DataLoader(
            dataset=gaze_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True)

        

        if not os.path.exists(evalpath):
            os.makedirs(evalpath)


        # list all epochs for testing
        folder = os.listdir(snapshot_path)
        folder.sort(key=natural_keys)
        softmax = nn.Softmax(dim=1)
        with open(os.path.join(evalpath,data_set+".log"), 'w') as outfile:
            configuration = f"\ntest configuration = gpu_id={gpu}, batch_size={batch_size}, model_arch={arch}\nStart testing dataset={data_set}----------------------------------------\n"
            print(configuration)
            outfile.write(configuration)
            epoch_list=[]
            avg_yaw=[]
            avg_pitch=[]
            avg_MAE=[]
            for epochs in folder:
                # Base network structure
                model=getArch(arch, 90)
                saved_state_dict = torch.load(os.path.join(snapshot_path, epochs))
                model.load_state_dict(saved_state_dict)
                model.cuda(gpu)
                model.eval()
                total = 0
                idx_tensor = [idx for idx in range(90)]
                idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
                avg_error = .0
                
                
                with torch.no_grad():           
                    for j, (images, labels, cont_labels, name) in enumerate(test_loader):
                        images = Variable(images).cuda(gpu)
                        total += cont_labels.size(0)

                        label_pitch = cont_labels[:,0].float()*np.pi/180
                        label_yaw = cont_labels[:,1].float()*np.pi/180
                        

                        gaze_pitch, gaze_yaw = model(images)
                        
                        # Binned predictions
                        _, pitch_bpred = torch.max(gaze_pitch.data, 1)
                        _, yaw_bpred = torch.max(gaze_yaw.data, 1)
                        
            
                        # Continuous predictions
                        pitch_predicted = softmax(gaze_pitch)
                        yaw_predicted = softmax(gaze_yaw)
                        
                        # mapping from binned (0 to 28) to angels (-180 to 180)  
                        pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 4 - 180
                        yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 4 - 180

                        pitch_predicted = pitch_predicted*np.pi/180
                        yaw_predicted = yaw_predicted*np.pi/180

                        for p,y,pl,yl in zip(pitch_predicted,yaw_predicted,label_pitch,label_yaw):
                            avg_error += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))
                        
        
                    
                x = ''.join(filter(lambda i: i.isdigit(), epochs))
                epoch_list.append(x)
                avg_MAE.append(avg_error/total)
                loger = f"[{epochs}---{args.dataset}] Total Num:{total},MAE:{avg_error/total}\n"
                outfile.write(loger)
                print(loger)
        
        fig = plt.figure(figsize=(14, 8))        
        plt.xlabel('epoch')
        plt.ylabel('avg')
        plt.title('Gaze angular error')
        plt.legend()
        plt.plot(epoch_list, avg_MAE, color='k', label='mae')
        fig.savefig(os.path.join(evalpath,data_set+".png"), format='png')
        plt.show()

 

    elif data_set=="mpiigaze":
        model_used=getArch(arch, bins)

        for fold in range(15):
            folder = os.listdir(args.gazeMpiilabel_dir)
            folder.sort()
            testlabelpathombined = [os.path.join(args.gazeMpiilabel_dir, j) for j in folder] 
            gaze_dataset=Mpiigaze(testlabelpathombined,args.gazeMpiimage_dir, transformations, False, angle, fold)

            test_loader = torch.utils.data.DataLoader(
                dataset=gaze_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True)
            
            
            if not os.path.exists(os.path.join(evalpath, f"fold"+str(fold))):
                os.makedirs(os.path.join(evalpath, f"fold"+str(fold)))

            # list all epochs for testing
            folder = os.listdir(os.path.join(snapshot_path,"fold"+str(fold)))
            folder.sort(key=natural_keys)
            
            softmax = nn.Softmax(dim=1)
            with open(os.path.join(evalpath, os.path.join("fold"+str(fold), data_set+".log")), 'w') as outfile:
                configuration = f"\ntest configuration equal gpu_id={gpu}, batch_size={batch_size}, model_arch={arch}\nStart testing dataset={data_set}, fold={fold}---------------------------------------\n"
                print(configuration)
                outfile.write(configuration)
                epoch_list=[]
                avg_MAE=[]
                for epochs in folder: 
                    model=model_used
                    saved_state_dict = torch.load(os.path.join(snapshot_path+"/fold"+str(fold),epochs))
                    model= nn.DataParallel(model,device_ids=[0])
                    model.load_state_dict(saved_state_dict)
                    model.cuda(gpu)
                    model.eval()
                    total = 0
                    idx_tensor = [idx for idx in range(28)]
                    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
                    avg_error = .0
                    with torch.no_grad():
                        for j, (images, labels, cont_labels, name) in enumerate(test_loader):
                            images = Variable(images).cuda(gpu)
                            total += cont_labels.size(0)

                            label_pitch = cont_labels[:,0].float()*np.pi/180
                            label_yaw = cont_labels[:,1].float()*np.pi/180
                            

                            gaze_pitch, gaze_yaw = model(images)
                            
                            # Binned predictions
                            _, pitch_bpred = torch.max(gaze_pitch.data, 1)
                            _, yaw_bpred = torch.max(gaze_yaw.data, 1)
                            
                
                            # Continuous predictions
                            pitch_predicted = softmax(gaze_pitch)
                            yaw_predicted = softmax(gaze_yaw)
                            
                            # mapping from binned (0 to 28) to angels (-42 to 42)                
                            pitch_predicted = \
                                torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 42
                            yaw_predicted = \
                                torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 42
                            
                            
                            pitch_predicted = pitch_predicted*np.pi/180
                            yaw_predicted = yaw_predicted*np.pi/180

                            for p,y,pl,yl in zip(pitch_predicted, yaw_predicted, label_pitch, label_yaw):
                                avg_error += angular(gazeto3d([p,y]), gazeto3d([pl,yl]))
            
                        
                    x = ''.join(filter(lambda i: i.isdigit(), epochs))
                    epoch_list.append(x)
                    avg_MAE.append(avg_error/ total)
                    loger = f"[{epochs}---{args.dataset}] Total Num:{total},MAE:{avg_error/total} \n"
                    outfile.write(loger)
                    print(loger)
        
            fig = plt.figure(figsize=(14, 8))        
            plt.xlabel('epoch')
            plt.ylabel('avg')
            plt.title('Gaze angular error')
            plt.legend()
            plt.plot(epoch_list, avg_MAE, color='k', label='mae')
            fig.savefig(os.path.join(evalpath, os.path.join("fold"+str(fold), data_set+".png")), format='png')
            # plt.show()

            
           
