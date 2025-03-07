
from PyQt5 import QtCore, QtGui, QtWidgets
import subprocess

class Ui_Face_Recognition_App(object):
    def setupUi(self, Face_Recognition_App):
        Face_Recognition_App.setObjectName("Face_Recognition_App")
        Face_Recognition_App.resize(1118, 667)
        self.centralwidget = QtWidgets.QWidget(Face_Recognition_App)
        self.centralwidget.setObjectName("centralwidget")
        self.Addface = QtWidgets.QPushButton(self.centralwidget)
        self.Addface.setGeometry(QtCore.QRect(30, 270, 111, 61))
        self.Addface.setObjectName("Addface")
        self.Recognition = QtWidgets.QPushButton(self.centralwidget)
        self.Recognition.setGeometry(QtCore.QRect(220, 270, 111, 61))
        self.Recognition.setObjectName("Recognition")
        self.Percentage = QtWidgets.QPushButton(self.centralwidget)
        self.Percentage.setGeometry(QtCore.QRect(400, 270, 111, 61))
        self.Percentage.setObjectName("Percentage")
        self.Object = QtWidgets.QPushButton(self.centralwidget)
        self.Object.setGeometry(QtCore.QRect(750, 270, 111, 61))
        self.Object.setObjectName("Object")
        self.Emotion = QtWidgets.QPushButton(self.centralwidget)
        self.Emotion.setGeometry(QtCore.QRect(570, 270, 111, 61))
        self.Emotion.setObjectName("Emotion")
        self.label_addface = QtWidgets.QLabel(self.centralwidget)
        self.label_addface.setGeometry(QtCore.QRect(30, 100, 101, 101))
        self.label_addface.setText("")
        self.label_addface.setPixmap(QtGui.QPixmap("addface.png"))
        self.label_addface.setScaledContents(True)
        self.label_addface.setObjectName("label_addface")
        self.label_recognition = QtWidgets.QLabel(self.centralwidget)
        self.label_recognition.setGeometry(QtCore.QRect(220, 100, 101, 101))
        self.label_recognition.setText("")
        self.label_recognition.setPixmap(QtGui.QPixmap("facerecog.png"))
        self.label_recognition.setScaledContents(True)
        self.label_recognition.setObjectName("label_recognition")
        self.label_percentage = QtWidgets.QLabel(self.centralwidget)
        self.label_percentage.setGeometry(QtCore.QRect(400, 100, 101, 101))
        self.label_percentage.setText("")
        self.label_percentage.setPixmap(QtGui.QPixmap("facerecog.webp"))
        self.label_percentage.setScaledContents(True)
        self.label_percentage.setObjectName("label_percentage")
        self.label_emotion = QtWidgets.QLabel(self.centralwidget)
        self.label_emotion.setGeometry(QtCore.QRect(560, 100, 101, 101))
        self.label_emotion.setText("")
        self.label_emotion.setPixmap(QtGui.QPixmap("emotion.png"))
        self.label_emotion.setScaledContents(True)
        self.label_emotion.setObjectName("label_emotion")
        self.Home = QtWidgets.QLabel(self.centralwidget)
        self.Home.setGeometry(QtCore.QRect(730, 100, 141, 101))
        self.Home.setText("")
        self.Home.setPixmap(QtGui.QPixmap("object.png"))
        self.Home.setScaledContents(True)
        self.Home.setObjectName("Home")
        Face_Recognition_App.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Face_Recognition_App)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1118, 21))
        self.menubar.setObjectName("menubar")
        self.menuAdd_Face = QtWidgets.QMenu(self.menubar)
        self.menuAdd_Face.setObjectName("menuAdd_Face")
        self.menuRecognition = QtWidgets.QMenu(self.menubar)
        self.menuRecognition.setObjectName("menuRecognition")
        self.menuPercentage = QtWidgets.QMenu(self.menubar)
        self.menuPercentage.setObjectName("menuPercentage")
        self.menuEmotion = QtWidgets.QMenu(self.menubar)
        self.menuEmotion.setObjectName("menuEmotion")
        self.menuObject_Detector = QtWidgets.QMenu(self.menubar)
        self.menuObject_Detector.setObjectName("menuObject_Detector")
        self.menuClose = QtWidgets.QMenu(self.menubar)
        self.menuClose.setObjectName("menuClose")
        self.menuHome = QtWidgets.QMenu(self.menubar)
        self.menuHome.setObjectName("menuHome")
        Face_Recognition_App.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Face_Recognition_App)
        self.statusbar.setObjectName("statusbar")
        Face_Recognition_App.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuHome.menuAction())
        self.menubar.addAction(self.menuAdd_Face.menuAction())
        self.menubar.addAction(self.menuRecognition.menuAction())
        self.menubar.addAction(self.menuPercentage.menuAction())
        self.menubar.addAction(self.menuEmotion.menuAction())
        self.menubar.addAction(self.menuObject_Detector.menuAction())
        self.menubar.addAction(self.menuClose.menuAction())

        self.retranslateUi(Face_Recognition_App)
        QtCore.QMetaObject.connectSlotsByName(Face_Recognition_App)

    def retranslateUi(self, Face_Recognition_App):
        _translate = QtCore.QCoreApplication.translate
        Face_Recognition_App.setWindowTitle(_translate("Face_Recognition_App", "Face Recognition Application"))
        self.Addface.setText(_translate("Face_Recognition_App", "Add Face"))
        self.Recognition.setText(_translate("Face_Recognition_App", "Recognition"))
        self.Percentage.setText(_translate("Face_Recognition_App", "Percentage"))
        self.Object.setText(_translate("Face_Recognition_App", "Object Detector"))
        self.Emotion.setText(_translate("Face_Recognition_App", "Emotion"))
        self.menuAdd_Face.setTitle(_translate("Face_Recognition_App", "Add Face"))
        self.menuRecognition.setTitle(_translate("Face_Recognition_App", "Recognition"))
        self.menuPercentage.setTitle(_translate("Face_Recognition_App", "Percentage"))
        self.menuEmotion.setTitle(_translate("Face_Recognition_App", "Emotion"))
        self.menuObject_Detector.setTitle(_translate("Face_Recognition_App", "Object Detector"))
        self.menuClose.setTitle(_translate("Face_Recognition_App", "Close"))
        self.menuHome.setTitle(_translate("Face_Recognition_App", "Home"))



        self.Addface.clicked.connect(self.on_add_face_clicked)
        self.Recognition.clicked.connect(self.on_recognition_clicked)
        self.Percentage.clicked.connect(self.on_percentage_clicked)
        self.Object.clicked.connect(self.on_object_detection_clicked)
        self.Emotion.clicked.connect(self.on_emotion_recognition_clicked)

        # Connect menu actions
        self.menuAdd_Face.triggered.connect(self.on_add_face_triggered)
        self.menuRecognition.triggered.connect(self.on_recognition_triggered)
        self.menuPercentage.triggered.connect(self.on_percentage_triggered)
        self.menuEmotion.triggered.connect(self.on_emotion_triggered)
        self.menuObject_Detector.triggered.connect(self.on_object_detection_triggered)
        self.menuClose.triggered.connect(self.on_close_triggered)
        self.menuHome.triggered.connect(self.on_home_triggered)

        # Existing setup code...

    # Button click event handlers
    def on_add_face_clicked(self):
        # Code to handle add face button click
        subprocess.Popen(["python", "Addface.py"])
        pass

    def on_recognition_clicked(self):
        # Code to handle recognition button click
        subprocess.Popen(["python", "recognition.py"])
        pass

    def on_percentage_clicked(self):
        # Code to handle percentage button click
        subprocess.Popen(["python", "percentage.py"])
        pass

    def on_object_detection_clicked(self):
        # Code to handle object detection button click
        subprocess.Popen(["python", "object.py"])

        pass

    def on_emotion_recognition_clicked(self):
        # Code to handle emotion recognition button click
        subprocess.Popen(["python", "emotion.py"])
        pass

    # Menu action event handlers
    def on_add_face_triggered(self):
        # Code to handle add face menu action
        subprocess.Popen(["python", "Addface.py"])

        pass

    def on_recognition_triggered(self):
        # Code to handle recognition menu action
        subprocess.Popen(["python", "recognition.py"])
        pass

    def on_percentage_triggered(self):
        # Code to handle percentage menu action
        subprocess.Popen(["python", "percentage.py"])
        pass

    def on_emotion_triggered(self):
        # Code to handle emotion menu action
        subprocess.Popen(["python", "emotion.py"])
        pass

    def on_object_detection_triggered(self):
        # Code to handle object detection menu action
        subprocess.Popen(["python", "object.py"])
        pass

    def on_close_triggered(self):
        # Code to handle close menu action
        sys.exit()
        pass

    def on_home_triggered(self):
        # Code to handle home menu action
        subprocess.Popen(["python", "face_recog.py"])
        pass



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Face_Recognition_App = QtWidgets.QMainWindow()
    ui = Ui_Face_Recognition_App()
    ui.setupUi(Face_Recognition_App)
    Face_Recognition_App.show()
    sys.exit(app.exec_())
