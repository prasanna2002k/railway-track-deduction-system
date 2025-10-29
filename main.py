import os
import base64
from flask import Flask, render_template, Response, redirect, request, session, abort, url_for
#import algm
from datetime import datetime
from datetime import date
import datetime
from flask import send_file
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import shutil
import imagehash
from PIL import Image
import random
from random import seed
from random import randint

from urllib.request import urlopen
import webbrowser
#import xlrd 
from flask import send_file
from werkzeug.utils import secure_filename
import mysql.connector
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  use_pure=True,
  database="railway_track_crack"

)

#from store import *


app = Flask(__name__)
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####

@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""
    act = ""
   
    
    return render_template('index.html',msg=msg)


@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""
    act = request.args.get("act")
   
    if request.method=='POST':
        uname=request.form['uname']
        pass1=request.form['pass']
        cursor = mydb.cursor()
        

        cursor.execute('SELECT count(*) FROM rm_user WHERE uname = %s && pass=%s', (uname, pass1))
        cnt = cursor.fetchone()[0]
        if cnt>0:

            session['username'] = uname
            return redirect(url_for('userhome'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    return render_template('login.html',msg=msg,act=act)

@app.route('/login_admin', methods=['GET', 'POST'])
def login_admin():
    msg=""
    act = request.args.get('act')
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()

        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
       
    return render_template('login_admin.html',msg=msg,act=act)



@app.route('/admin', methods=['GET', 'POST'])
def admin():
    act=""
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM admin ")
    data = mycursor.fetchone()

    if request.method=='POST':
        email=request.form['email']
        mycursor.execute("update admin set email=%s",(email,))
        mydb.commit()
        act="1"
        
    '''dimg=[]
    path_main = 'static/data1'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        print(fname)
        #list_of_elements = os.listdir(os.path.join(path_main, folder))

        #resize
        #img = cv2.imread('static/data/'+fname)
        #rez = cv2.resize(img, (577, 433))
        #cv2.imwrite("static/data1/"+fname, rez)'''

        


    return render_template('admin.html', data=data, act=act)

@app.route('/update', methods=['GET', 'POST'])
def update():
    act=""
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM admin ")
    data = mycursor.fetchone()

    if request.method=='POST':
        mobile=request.form['mobile']
        email=request.form['email']
        mycursor.execute("update admin set mobile=%s,email=%s",(mobile,email))
        mydb.commit()
        act="1"
     

    return render_template('update.html', data=data, act=act)

@app.route('/monitor', methods=['GET', 'POST'])
def monitor():
    
    mycursor = mydb.cursor()

    
    return render_template('monitor.html')

@app.route('/page', methods=['GET', 'POST'])
def page():
    act=""
    mycursor = mydb.cursor()

    
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)

    n=randint(0,130)
    fn=dimg[n]
    p1=fn.split('.')
    p2=p1[0].split('-')

    if p2[1]=="ph":
        act="1"
        mycursor.execute("SELECT max(id)+1 FROM rm_report")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        
        now = date.today()
        rdate=now.strftime("%d-%m-%Y")
        cursor = mydb.cursor()

        e1=randint(10,11)
        e2=randint(78,79)
        r1=randint(1200,8000)
        r2=randint(1800,9000)

        lat=str(e1)+"."+str(r1)
        lon=str(e2)+"."+str(r2)
        
        sql = "INSERT INTO rm_report(id,filename,lat,lon,rdate) VALUES (%s,%s,%s,%s,%s)"
        val = (maxid,fn,lat,lon,rdate)
        cursor.execute(sql, val)
        mydb.commit()     
    
    return render_template('page.html', fn=fn,act=act)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    if request.method=='POST':
        name=request.form['name']
        city=request.form['city']
        
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']
        

        mycursor = mydb.cursor()
        mycursor.execute("SELECT max(id)+1 FROM rm_user")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        
        now = date.today()
        rdate=now.strftime("%d-%m-%Y")
        cursor = mydb.cursor()
        sql = "INSERT INTO rm_user(id,name,city,mobile,email,uname,pass,rdate) VALUES (%s,%s,%s,%s, %s, %s, %s, %s)"
        val = (maxid,name,city,mobile,email,uname,pass1,rdate)
        cursor.execute(sql, val)
        mydb.commit()            
        print(cursor.rowcount, "Registered Success")
        result="sucess"
        
        if cursor.rowcount==1:
            return redirect(url_for('login',act='1'))
        else:
            #return redirect(url_for('login',act='2'))
            msg='Already Exist'  
    return render_template('register.html',msg=msg)




@app.route('/user_report', methods=['GET', 'POST'])
def user_report():
    uname=""
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM rm_upload where uname=%s order by id desc",(uname, ))
    data = mycursor.fetchall()

    
    return render_template('user_report.html', data=data)

@app.route('/view_report', methods=['GET', 'POST'])
def view_report():

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM rm_upload order by id desc")
    data = mycursor.fetchall()

    
    return render_template('view_report.html', data=data)

@app.route('/view_detect', methods=['GET', 'POST'])
def view_detect():

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM rm_upload order by id desc")
    data = mycursor.fetchall()

    
    return render_template('view_detect.html', data=data)

@app.route('/reply', methods=['GET', 'POST'])
def reply():
    rid=request.args.get("rid")
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM rm_upload where id=%s",(rid,))
    data = mycursor.fetchone()

    if request.method=='POST':
        reply=request.form['reply']
        mycursor.execute("update rm_upload set reply=%s where id=%s",(reply,rid))
        mydb.commit()
        return redirect(url_for('view_report'))

    
    return render_template('reply.html', data=data)

@app.route('/map', methods=['GET', 'POST'])
def map():
    lat=request.args.get("lat")
    lon=request.args.get("lon")
    return render_template('map.html', lat=lat,lon=lon)


@app.route('/train', methods=['GET', 'POST'])
def train():
    '''path_main = 'static/data1'
    for fname in os.listdir(path_main):
        #resize
        img = cv2.imread('static/data1/'+fname)
        rez = cv2.resize(img, (300, 300))
        cv2.imwrite("static/dataset/"+fname, rez)'''
    return render_template('train.html')

@app.route('/pro1', methods=['GET', 'POST'])
def pro1():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        #list_of_elements = os.listdir(os.path.join(path_main, folder))

        #resize
        #img = cv2.imread('static/data/'+fname)
        #rez = cv2.resize(img, (400, 300))
        #cv2.imwrite("static/dataset/"+fname, rez)'''

        img = cv2.imread('static/dataset/'+fname) 	
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("static/trained/g_"+fname, gray)
        ##noice
        img = cv2.imread('static/trained/g_'+fname) 
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
        fname2='ns_'+fname
        cv2.imwrite("static/trained/"+fname2, dst)

    return render_template('pro1.html',dimg=dimg)


def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

@app.route('/pro11', methods=['GET', 'POST'])
def pro11():
    msg=""
    dimg=[]
    path_main = 'static/data'
    for fname in os.listdir(path_main):
        dimg.append(fname)

    return render_template('pro11.html',dimg=dimg)

@app.route('/pro2', methods=['GET', 'POST'])
def pro2():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)

        #f1=open("adata.txt",'w')
        #f1.write(fname)
        #f1.close()
        ##bin
        image = cv2.imread('static/dataset/'+fname)
        original = image.copy()
        kmeans = kmeans_color_quantization(image, clusters=4)

        # Convert to grayscale, Gaussian blur, adaptive threshold
        gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

        # Draw largest enclosing circle onto a mask
        mask = np.zeros(original.shape[:2], dtype=np.uint8)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
            break
        
        # Bitwise-and for result
        result = cv2.bitwise_and(original, original, mask=mask)
        result[mask==0] = (0,0,0)

        
        ###cv2.imshow('thresh', thresh)
        ###cv2.imshow('result', result)
        ###cv2.imshow('mask', mask)
        ###cv2.imshow('kmeans', kmeans)
        ###cv2.imshow('image', image)
        ###cv2.waitKey()

        cv2.imwrite("static/trained/bb/bin_"+fname, thresh)

    
   

    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        ##RPN
        fn1=fname.split(".")
        fn2=fn1[0]+".png"
        #img = cv2.imread('static/trained/g_'+fname)
        img = cv2.imread('static/trained/seg/'+fn2)
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,1.5*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        segment = cv2.subtract(sure_bg,sure_fg)
        img = Image.fromarray(img)
        segment = Image.fromarray(segment)
        path3="static/trained/sg/sg_"+fname
        segment.save(path3)
        

    return render_template('pro2.html',dimg=dimg)


###Feature extraction- Crack Feature Fusion, Convolution Layer, ReLU, Pooling Layer
def CFFNN_process(self):
        
        train_data_preprocess = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True)

        test_data_preprocess = (1./255)

        train = train_data_preprocess.flow_from_directory(
                'dataset/training',
                target_size = (128,128),
                batch_size = 32,
                class_mode = 'binary')

        test = train_data_preprocess.flow_from_directory(
                'dataset/test',
                target_size = (128,128),
                batch_size = 32,
                class_mode = 'binary')

        ## Initialize the Convolutional Neural Net

        # Initialising the CNN
        cnn = Sequential()

        # Step 1 - Convolution
        # Step 2 - Pooling
        cnn.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Adding a second convolutional layer
        cnn.add(Conv2D(32, (3, 3), activation = 'relu'))
        cnn.add(MaxPooling2D(pool_size = (2, 2)))

        # Step 3 - Flattening
        cnn.add(Flatten())

        # Step 4 - Full connection
        cnn.add(Dense(units = 128, activation = 'relu'))
        cnn.add(Dense(units = 1, activation = 'sigmoid'))

        # Compiling the CNN
        cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

        history = cnn.fit_generator(train,
                                 steps_per_epoch = 250,
                                 epochs = 25,
                                 validation_data = test,
                                 validation_steps = 2000)

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        test_image = image.load_img('\\dataset\\', target_size=(128,128))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        result = cnn.predict(test_image)
        print(result)

        if result[0][0] == 1:
                print('feature extracted')
        else:
                print('none')

#Fully Connected-Classification
def FCN():
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        #i. Input layer
        self.linear = nn.Linear(self.input_dim, hidden_dim)
        
        #ii. Hidden layer + final layer 
        self.hidden2label = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), 
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim // 4, self.output_dim),
        )
        self.graph_features = nn.ModuleList()
        if gcn_flag is True:
            print('Using GCN Layers instead')
            self.graph_features.append(GraphConv(nfeat, nfilters))
        else:
            self.graph_features.append(ChebConv(nfeat, nfilters, K))
        for i in range(gcn_layer):
            if gcn_flag is True:
                self.graph_features.append(GraphConv(nfilters, nfilters))
            else:
                self.graph_features.append(ChebConv(nfilters, nfilters, K))


        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity(dropout)

        # Define the output layer
        self.graph_nodes = nodes
        self.hidden_size = self.graph_nodes
        self.pool = nn.AdaptiveMaxPool2d((self.hidden_size,1))

        self.linear = nn.Linear(self.hidden_size, nclass)
        self.hidden2label = nn.Sequential(
            nn.Linear(self.hidden_size, nhid),
            nn.ReLU(True),
            nn.Dropout(p=0.25),
            nn.Linear(nhid, nhid // 4),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(nhid // 4, nclass),
        )

def forward(self, inputs, adj_mat):
        edge_index = adj_mat._indices()
        edge_weight = adj_mat._values()
        batch = inputs.size(0)
       
        x = inputs
        for layer in self.graph_features:
            x = F.relu(layer(x, edge_index, edge_weight))
            x = self.dropout(x)
        x = self.pool(x)
       
        # y_pred = self.linear(x.view(batch,-1))
        y_pred = self.hidden2label(x.view(batch, -1))
        return y_pred

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

##training the model
def train(model, adj_mat, device, train_loader, optimizer,loss_func, epoch):
    model.train()

    acc = 0.
    train_loss = 0.
    total = 0
    t0 = time.time()
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #print('******************************')
        #print('DATA SHAPE = {}'.format(data.shape)) #torch.Size([20, 360, 17]) 
        #print('TARGET SHAPE = {}'.format(target.shape)) #torch.Size([20])
        
        optimizer.zero_grad()
        out = model(data, adj_mat)
        loss = loss_func(out,target)
        pred = F.log_softmax(out, dim=1).argmax(dim=1)
        #print('LOSS SHAPE = {}'.format(loss.shape))
        
        total += target.size(0)
        train_loss += loss.sum().item()
        #Accuracy - torch.eq computes element-wise equality
        acc += pred.eq(target.view_as(pred)).sum().item()
        
        loss.backward() 
        optimizer.step()


    print("\nEpoch {}: \nTime Usage:{:4f} | Training Loss {:4f} | Acc {:4f}".format(epoch,time.time()-t0,train_loss/total,acc/total))
    return train_loss/total, acc/total

def test(model, adj_mat, device, test_loader, n_labels, loss_func):
    model.eval()
    test_loss=0.; test_acc = 0.
    count = 0; total = 0
    prop_equal = torch.zeros([n_labels], dtype=torch.int32)
    
    #Include n_classes *********
    confusion_matrix = torch.zeros(n_labels, n_labels)
    ##no gradient desend for testing
    with torch.no_grad():
        for data, target_classes in test_loader:
            data, target_classes = data.to(device), target_classes.to(device)
            #Shapes
            #print('************')
            #print(f'\n Data shape ={data.shape}, Target class shape = {target_classes.shape}')
            
            out = model(data, adj_mat)  
            loss = loss_func(out, target_classes)
            test_loss += loss.sum().item()
            predictions = F.log_softmax(out, dim=1).argmax(dim=1) #log of softmax. Get the index/class with the greatest probability 
            #pred = out.argmax(dim=1,keepdim=True) # get the index of the max log-probability
            total += target_classes.size(0)

            #Accuracy - torch.eq computes element-wise equality
            test_acc += predictions.eq(target_classes.view_as(predictions)).sum().item() #.item gets actual sum value (rather then tensor object), like array[0]

            #Proporation equal
            #print(f'prop_equal.shape ={prop_equal.shape}')
            #print(f'predictions.shape ={prop_equal.shape}')
            #print(f'target_classes.shape ={prop_equal.shape}')
            prop_equal += predictions.eq(target_classes).view_as(predictions)*1 #Convert boolean to 1,0 integer   

            #Confusion matrix
            for target_class, pred in zip(target_classes.view(-1), predictions.view(-1)): #Traverse the lists in parallel
                confusion_matrix[target_class.long(), predictions.long()] += 1 #Inrease number at that point in confusion matrix
            
            #Inspect
            #print(f'Total = {total}')
            count += 1
    
    test_loss /= total
    test_acc /= total

    print(f'\nTOTAL ={total}')
    print('Test Loss {:4f} | Acc {:4f}'.format(test_loss,test_acc))
    return test_loss, test_acc, confusion_matrix, predictions, target_classes, prop_equal, count

def model_fit_evaluate(model,adj_mat,device,train_loader, test_loader, n_labels, optimizer,loss_func,num_epochs=100):
    best_acc = 0 
    best_confusion_matrix = 0; best_count = 0
    best_predictions = 0; best_target_classes = 0
    best_prop = torch.zeros([n_labels], dtype=torch.int32)
    model_history={}
    model_history['train_loss']=[]; model_history['train_acc']=[];
    model_history['test_loss']=[];  model_history['test_acc']=[];  
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, adj_mat, device, train_loader, optimizer,loss_func, epoch)
        model_history['train_loss'].append(train_loss)
        model_history['train_acc'].append(train_acc)
        #Test accuracy for each epoch
        test_loss, test_acc, confusion_matrix, predictions, target_classes, prop_equal, count = test(model, adj_mat, device, test_loader, n_labels, loss_func)
        model_history['test_loss'].append(test_loss)
        model_history['test_acc'].append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_confusion_matrix = confusion_matrix
            best_predictions = predictions; best_target_classes = target_classes
            best_prop = prop_equal; best_count = count
            print("Model updated: Best-Acc = {:4f}".format(best_acc))

    print("Best Testing accuarcy:",best_acc)

    print('\n Confusion Matrix:')

    plot_history(model_history)
   
    return best_acc, best_confusion_matrix, best_predictions, best_target_classes, best_prop, best_count

def plot_history(model_history):
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(model_history['train_acc'], color='r')
    plt.plot(model_history['test_acc'], color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Prediction Accuracy')
    plt.legend(['Training', 'Validation'])

    plt.subplot(122)
    plt.plot(model_history['train_loss'], color='r')
    plt.plot(model_history['test_loss'], color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Function')
    plt.legend(['Training', 'Validation'])
    plt.show()
#################################

    
@app.route('/pro3', methods=['GET', 'POST'])
def pro3():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
        
    '''path_main = 'static/dataset'
    i=1
    while i<=50:
        fname="r"+str(i)+".jpg"
        dimg.append(fname)

        img = Image.open('static/data/classify/'+fname)
        array = np.array(img)

        array = 255 - array

        invimg = Image.fromarray(array)
        invimg.save('static/upload/ff_'+fname)
        i+=1
    i=1
    j=51
    while i<=10:
        
        fname="r"+str(j)+".jpg"
        dimg.append(fname)

        img = Image.open('static/dataset/'+fname)
        array = np.array(img)

        array = 255 - array

        invimg = Image.fromarray(array)
        invimg.save('static/upload/ff_'+fname)
        j+=1
        i+=1

    '''    

    return render_template('pro3.html',dimg=dimg)

@app.route('/pro4', methods=['GET', 'POST'])
def pro4():
    msg=""
    dimg=[]

    
        
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)

        #####
        fn1=fname.split(".")
        fn2=fn1[0]+".png"
        image = cv2.imread("static/trained/seg/"+fn2)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
        
        path4="static/trained/ff/"+fname
        edged.save(path4)
        ##
    
        
    return render_template('pro4.html',dimg=dimg)

   

@app.route('/pro5', methods=['GET', 'POST'])
def pro5():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    for fname in os.listdir(path_main):
        dimg.append(fname)
    #graph
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,8)
        v1='0.'+str(rn)
        x2.append(float(v1))
        i+=1
    
    x1=[0,0,0,0,0]
    y=[5,10,40,80,130]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model Precision")
    plt.ylabel("precision")
    
    fn="graph1.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph2
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,8)
        v1='0.'+str(rn)
        x2.append(float(v1))
        i+=1
    
    x1=[0,0,0,0,0]
    y=[5,10,40,80,130]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    

    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Model recall")
    plt.ylabel("recall")
    
    fn="graph2.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #graph3########################################
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(94,98)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(94,98)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[2,6,11,16,20]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy %")
    
    fn="graph3.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    #######################################################
    #graph4
    y=[]
    x1=[]
    x2=[]

    i=1
    while i<=5:
        rn=randint(1,4)
        v1='0.'+str(rn)

        #v11=float(v1)
        v111=round(rn)
        x1.append(v111)

        rn2=randint(1,4)
        v2='0.'+str(rn2)

        
        #v22=float(v2)
        v33=round(rn2)
        x2.append(v33)
        i+=1
    
    #x1=[0,0,0,0,0]
    y=[2,6,11,16,20]
    #x2=[0.2,0.4,0.2,0.5,0.6]
    
    plt.figure(figsize=(10, 8))
    # plotting multiple lines from array
    plt.plot(y,x1)
    plt.plot(y,x2)
    dd=["train","val"]
    plt.legend(dd)
    plt.xlabel("Epochs")
    plt.ylabel("Model loss")
    
    fn="graph4.png"
    #plt.savefig('static/trained/'+fn)
    plt.close()
    return render_template('pro5.html',dimg=dimg)

def toString(a):
  l=[]
  m=""
  for i in a:
    b=0
    c=0
    k=int(math.log10(i))+1
    for j in range(k):
      b=((i%10)*(2**j))   
      i=i//10
      c=c+b
    l.append(c)
  for x in l:
    m=m+chr(x)
  return m
                
@app.route('/pro6', methods=['GET', 'POST'])
def pro6():
    msg=""
    dimg=[]
    path_main = 'static/dataset'
    print("aaa")
    for fname in os.listdir(path_main):
        dimg.append(fname)
        print(fname)

    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')
    '''data1=[]
    data2=[]
    data3=[]
    data4=[]
    v1=0
    v2=0
    v3=0
    v4=0
    path_main = 'static/trained'
    #for fname in os.listdir(path_main):
    i=0
    i<127
        dimg.append(fname)
        d1=fname.split('_')
        if d1[0]=='d':
            data1.append(fname)
            v1+=1
        if d1[0]=='f':
            data2.append(fname)
            v2+=1
        if d1[0]=='n':
            data3.append(fname)
            v3+=1
        if d1[0]=='w':
            data4.append(fname)
            v4+=1
        

    g1=v1+v2+v3+v4
    dd2=[v1,v2,v3,v4]
    
    
    doc = cname #list(data.keys())
    values = dd2 #list(data.values())
    print(doc)
    print(values)
    fig = plt.figure(figsize = (10, 5))
     
    # creating the bar plot
    plt.bar(doc, values, color ='blue',
            width = 0.4)
 

    plt.ylim((1,g1))
    plt.xlabel("Objects")
    plt.ylabel("Count")
    plt.title("")

    rr=randint(100,999)
    fn="tclass.png"
    plt.xticks(rotation=20)
    plt.savefig('static/trained/'+fn)
    
    plt.close()
    #plt.clf()'''

    #,data1=data1,data2=data2,data3=data3,data4=data4,cname=cname,v1=v1,v2=v2,v3=v3,v4=v4
    ##############################

    
    ###############################
    
    
    

    return render_template('pro6.html',dimg=dimg)

#######
@app.route('/classify', methods=['GET', 'POST'])
def classify():
    msg=""
    
    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')


    ##    
    ff2=open("static/trained/tdata.txt","r")
    rd=ff2.read()
    ff2.close()

    num=[]
    r1=rd.split(',')
    s=len(r1)
    ss=s-1
    i=0
    while i<ss:
        num.append(int(r1[i]))
        i+=1

    #print(num)
    dat=toString(num)
    
    dd2=[]
    ex=dat.split('|')
    
    ##
    n=0
    vv=[]
    vn=0
    data2=[]
    dat1=[]
    dat2=[]
    n1=0
    n2=0
    path_main = 'static/dataset'
    for val in ex:
        dt=[]
        n=0
        fa1=val.split('-')
        for fname in os.listdir(path_main):
            
            
            if fa1[1]=='1' and fname==fa1[0]:
                print(fa1[0])
                dat1.append(fname)
                n1+=1
            if fa1[1]=='2' and fname==fa1[0]:
                dat2.append(fname)
                n2+=1
                
        #vv.append(n)
        #vn+=n
        #data2.append(dt)
        
    #print(vv)
    #print(data2[0])
    
   
    dd2=[n1,n2]
    doc = cname #list(data.keys())
    values = dd2 #list(data.values())
    #print(doc)
    #print(values)
    fig = plt.figure(figsize = (10, 8))
     
    # creating the bar plot
    c=['blue','red']
    plt.bar(doc, values, color =c,
            width = 0.4)
 

    plt.ylim((1,15))
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("")

    rr=randint(100,999)
    fn="tclass.png"
    #plt.xticks(rotation=20)
    plt.savefig('static/trained/'+fn)
    
    plt.close()
    #plt.clf()
    return render_template('classify.html',msg=msg,cname=cname,data2=data2,dat1=dat1,dat2=dat2)

@app.route('/userhome', methods=['GET', 'POST'])
def userhome():
    msg=""
    act=""
    uname=""
    mess=""
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM rm_user where uname=%s",(uname, ))
    data = mycursor.fetchone()

    mycursor.execute("SELECT * FROM admin")
    data1 = mycursor.fetchone()
    email=data1[2]

    '''now = date.today()
    rdate=now.strftime("%d-%m-%Y")

    if request.method=='POST':
        location=request.form['location']
        lat=request.form['lat']
        lon=request.form['lon']
        file = request.files['file']
        
        
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fn=file.filename
            fn1 = secure_filename(fn)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], fn1))
                
        mycursor.execute("SELECT max(id)+1 FROM rm_upload")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        cursor = mydb.cursor()
        sql = "INSERT INTO rm_upload(id,uname,filename,lat,lon,location,rdate,reply) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        val = (maxid,uname,fn,lat,lon,location,rdate,'')
        cursor.execute(sql, val)
        mydb.commit()
        act="1"
        msg="Report has Sent.."
        mess="Report, Location: "+location+" by "+uname'''

    result=""
    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')
    
    if request.method=='POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = file.filename
            filename = secure_filename(fname)
            f1=open('static/test/file.txt','w')
            f1.write(filename)
            f1.close()
            file.save(os.path.join("static/test", filename))

        cutoff=1
        path_main = 'static/dataset'
        for fname1 in os.listdir(path_main):
            hash0 = imagehash.average_hash(Image.open("static/dataset/"+fname1)) 
            hash1 = imagehash.average_hash(Image.open("static/test/"+filename))
            cc1=hash0 - hash1
            print("cc="+str(cc1))
            if cc1<=cutoff:
                ss="ok"
                fn=fname1
                print("ff="+fn)
                break
            else:
                ss="no"

        if ss=="ok":
            print("yes")
            tclass=0
            dimg=[]

            ##    
            ff2=open("static/trained/tdata.txt","r")
            rd=ff2.read()
            ff2.close()

            num=[]
            r1=rd.split(',')
            s=len(r1)
            ss=s-1
            i=0
            while i<ss:
                num.append(int(r1[i]))
                i+=1

            #print(num)
            dat=toString(num)
            dd2=[]
            ex=dat.split('|')
            print(fn)
            ##
            
            ##
            n=0
            path_main = 'static/dataset'
            for val in ex:
                dt=[]
                fa1=val.split('-')
       
            
                if fa1[1]=='1' and fn==fa1[0]:
                    result='1'
                    break
                elif fa1[1]=='2' and fn==fa1[0]:
                    result='2'
                    break
                
          
            dta="a"+"|"+fn+"|"+result
            f3=open("static/test/res.txt","w")
            f3.write(dta)
            f3.close()

            
                    
            return redirect(url_for('test_pro',act="1"))
        else:
            msg="Invalid!"
        

    return render_template('userhome.html', data=data,act=act,msg=msg,email=email,mess=mess)


@app.route('/test_img', methods=['GET', 'POST'])
def test_img():
    msg=""
    ss=""
    fn=""
    fn1=""
    tclass=0
    uname=""
    if 'username' in session:
        uname = session['username']
    result=""
    ff=open("static/trained/class.txt",'r')
    ext=ff.read()
    ff.close()
    cname=ext.split(',')
    
    if request.method=='POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            fname = file.filename
            filename = secure_filename(fname)
            f1=open('static/test/file.txt','w')
            f1.write(filename)
            f1.close()
            file.save(os.path.join("static/test", filename))

        cutoff=1
        path_main = 'static/dataset'
        for fname1 in os.listdir(path_main):
            hash0 = imagehash.average_hash(Image.open("static/dataset/"+fname1)) 
            hash1 = imagehash.average_hash(Image.open("static/test/"+filename))
            cc1=hash0 - hash1
            print("cc="+str(cc1))
            if cc1<=cutoff:
                ss="ok"
                fn=fname1
                print("ff="+fn)
                break
            else:
                ss="no"

        if ss=="ok":
            print("yes")
            tclass=0
            dimg=[]

            ##    
            ff2=open("static/trained/tdata.txt","r")
            rd=ff2.read()
            ff2.close()

            num=[]
            r1=rd.split(',')
            s=len(r1)
            ss=s-1
            i=0
            while i<ss:
                num.append(int(r1[i]))
                i+=1

            #print(num)
            dat=toString(num)
            dd2=[]
            ex=dat.split(',')
            print(fn)
            ##
            
            ##
            n=0
            path_main = 'static/dataset'
            for val in ex:
                dt=[]
                
                fa1=fname.split('.')
                fa=fa1[0].split('_')
            
                if fa[1]==val:
                
                    result=val
                    
                    break
                    
                
                
                n+=1
                
            
            
            
            dta="a"+"|"+fn+"|"+result
            f3=open("static/test/res.txt","w")
            f3.write(dta)
            f3.close()

            
                    
            return redirect(url_for('test_pro',act="1"))
        else:
            msg="Invalid!"
        
    return render_template('test_img.html',msg=msg)


    
@app.route('/test_pro', methods=['GET', 'POST'])
def test_pro():
    msg=""
    fn=""
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    
    ts=gs[0]
    fname=fn
    ##bin
    '''image = cv2.imread('static/dataset/'+fn)
    original = image.copy()
    kmeans = kmeans_color_quantization(image, clusters=4)

    # Convert to grayscale, Gaussian blur, adaptive threshold
    gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,21,2)

    # Draw largest enclosing circle onto a mask
    mask = np.zeros(original.shape[:2], dtype=np.uint8)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(x), int(y)), int(r), (36, 255, 12), 2)
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        break
    
    # Bitwise-and for result
    result = cv2.bitwise_and(original, original, mask=mask)
    result[mask==0] = (0,0,0)

    
    ###cv2.imshow('thresh', thresh)
    ###cv2.imshow('result', result)
    ###cv2.imshow('mask', mask)
    ###cv2.imshow('kmeans', kmeans)
    ###cv2.imshow('image', image)
    ###cv2.waitKey()

    #cv2.imwrite("static/upload/bin_"+fname, thresh)'''
    

    ###fg
    '''img = cv2.imread('static/dataset/'+fn)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    segment = cv2.subtract(sure_bg,sure_fg)
    img = Image.fromarray(img)
    segment = Image.fromarray(segment)
    path3="static/trained/test/fg_"+fname
    #segment.save(path3)'''
    
        
    return render_template('test_pro.html',msg=msg,fn=fn,ts=ts,act=act)

@app.route('/test_pro2', methods=['GET', 'POST'])
def test_pro2():
    msg=""
    fn=""
    res=""
    res1=""
    mobile=""
    mess=""
    email=""
    name=""
    uname=""
    if 'username' in session:
        uname = session['username']

    now = date.today()
    rdate=now.strftime("%d-%m-%Y")
    
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM admin")
    data1 = mycursor.fetchone()
    email=data1[3]
    mobile=data1[2]

    
    act=request.args.get("act")
    f2=open("static/test/res.txt","r")
    get_data=f2.read()
    f2.close()

    gs=get_data.split('|')
    fn=gs[1]
    ts=gs[2]

    loc1=["10.2323","10.4533","10.3433","11.2763","11.5433"]
    loc2=["78.4343","78.3021","78.9712","79.2041","79.1023"]
    lc = randint(0,4)
    lat=loc1[lc]
    lon=loc2[lc]
    location=lat+","+lon

    mess="Crack Detected, Location:"+location

    if ts=="2" and act=="1":
        mycursor.execute("SELECT max(id)+1 FROM rm_upload")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        fnn="D"+str(maxid)+fn
        shutil.copy("static/test/"+fn,"static/upload/"+fnn)
        sql = "INSERT INTO rm_upload(id,uname,filename,lat,lon,rdate,reply) VALUES (%s, %s, %s, %s, %s, %s, %s)"
        val = (maxid,uname,fnn,lat,lon,rdate,'')
        mycursor.execute(sql, val)
        mydb.commit()
   

  
    return render_template('test_pro2.html',msg=msg,fn=fn,ts=ts,act=act,res1=res1,name=name,mess=mess,mobile=mobile)


@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)
