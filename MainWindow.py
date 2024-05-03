# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'MainWindow.ui'
##
## Created by: Qt User Interface Compiler version 6.7.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QDoubleSpinBox, QFrame, QGraphicsView,
    QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QMenu, QMenuBar, QPushButton, QSizePolicy,
    QStatusBar, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(809, 613)
        self.action_download_img = QAction(MainWindow)
        self.action_download_img.setObjectName(u"action_download_img")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_3 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout = QGridLayout(self.frame)
        self.gridLayout.setObjectName(u"gridLayout")
        self.dspinbox_region_growing = QDoubleSpinBox(self.frame)
        self.dspinbox_region_growing.setObjectName(u"dspinbox_region_growing")
        self.dspinbox_region_growing.setDecimals(0)
        self.dspinbox_region_growing.setMinimum(5.000000000000000)
        self.dspinbox_region_growing.setMaximum(100.000000000000000)

        self.gridLayout.addWidget(self.dspinbox_region_growing, 3, 1, 1, 1)

        self.dspinbox_k_means = QDoubleSpinBox(self.frame)
        self.dspinbox_k_means.setObjectName(u"dspinbox_k_means")
        self.dspinbox_k_means.setDecimals(0)
        self.dspinbox_k_means.setMinimum(2.000000000000000)
        self.dspinbox_k_means.setMaximum(50.000000000000000)

        self.gridLayout.addWidget(self.dspinbox_k_means, 0, 1, 1, 1)

        self.dspinbox_DBSCAN = QDoubleSpinBox(self.frame)
        self.dspinbox_DBSCAN.setObjectName(u"dspinbox_DBSCAN")
        self.dspinbox_DBSCAN.setDecimals(0)
        self.dspinbox_DBSCAN.setMinimum(50.000000000000000)
        self.dspinbox_DBSCAN.setMaximum(500.000000000000000)

        self.gridLayout.addWidget(self.dspinbox_DBSCAN, 1, 1, 1, 1)

        self.dspinbox_growing_seed = QDoubleSpinBox(self.frame)
        self.dspinbox_growing_seed.setObjectName(u"dspinbox_growing_seed")
        self.dspinbox_growing_seed.setDecimals(0)
        self.dspinbox_growing_seed.setMinimum(20.000000000000000)
        self.dspinbox_growing_seed.setMaximum(300.000000000000000)

        self.gridLayout.addWidget(self.dspinbox_growing_seed, 2, 1, 1, 1)

        self.btn_watershed = QPushButton(self.frame)
        self.btn_watershed.setObjectName(u"btn_watershed")
        font = QFont()
        font.setFamilies([u"Arial"])
        font.setPointSize(10)
        self.btn_watershed.setFont(font)

        self.gridLayout.addWidget(self.btn_watershed, 4, 0, 1, 1)

        self.btn_region_growin = QPushButton(self.frame)
        self.btn_region_growin.setObjectName(u"btn_region_growin")
        self.btn_region_growin.setFont(font)

        self.gridLayout.addWidget(self.btn_region_growin, 3, 0, 1, 1)

        self.btn_growing_seed = QPushButton(self.frame)
        self.btn_growing_seed.setObjectName(u"btn_growing_seed")
        self.btn_growing_seed.setFont(font)

        self.gridLayout.addWidget(self.btn_growing_seed, 2, 0, 1, 1)

        self.btn_k_means = QPushButton(self.frame)
        self.btn_k_means.setObjectName(u"btn_k_means")
        self.btn_k_means.setFont(font)

        self.gridLayout.addWidget(self.btn_k_means, 0, 0, 1, 1)

        self.btn_DBSCAN = QPushButton(self.frame)
        self.btn_DBSCAN.setObjectName(u"btn_DBSCAN")
        self.btn_DBSCAN.setFont(font)

        self.gridLayout.addWidget(self.btn_DBSCAN, 1, 0, 1, 1)


        self.verticalLayout.addWidget(self.frame)

        self.frame_3 = QFrame(self.centralwidget)
        self.frame_3.setObjectName(u"frame_3")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setMaximumSize(QSize(500, 300))
        self.frame_3.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_3.setFrameShadow(QFrame.Shadow.Raised)
        self.gridLayout_2 = QGridLayout(self.frame_3)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.graphicsView = QGraphicsView(self.frame_3)
        self.graphicsView.setObjectName(u"graphicsView")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy1)
        self.graphicsView.setMaximumSize(QSize(50, 50))

        self.gridLayout_2.addWidget(self.graphicsView, 0, 1, 1, 1)

        self.label = QLabel(self.frame_3)
        self.label.setObjectName(u"label")
        self.label.setMaximumSize(QSize(16777215, 100))

        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)

        self.btn_drop_color = QPushButton(self.frame_3)
        self.btn_drop_color.setObjectName(u"btn_drop_color")

        self.gridLayout_2.addWidget(self.btn_drop_color, 1, 0, 1, 1)


        self.verticalLayout.addWidget(self.frame_3)


        self.horizontalLayout_3.addLayout(self.verticalLayout)

        self.frame_2 = QFrame(self.centralwidget)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setFrameShape(QFrame.Shape.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Shadow.Raised)
        self.horizontalLayout_2 = QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.graph_view_img = QGraphicsView(self.frame_2)
        self.graph_view_img.setObjectName(u"graph_view_img")

        self.horizontalLayout_2.addWidget(self.graph_view_img)


        self.horizontalLayout_3.addWidget(self.frame_2)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 809, 22))
        self.menu = QMenu(self.menubar)
        self.menu.setObjectName(u"menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menu.menuAction())
        self.menu.addAction(self.action_download_img)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"\u041b\u0430\u0431\u043e\u0440\u0430\u0442\u043e\u0440\u043d\u0430\u044f \u0440\u0430\u0431\u043e\u0442\u0430 5 \u041f\u0440\u043e\u0449\u0435\u043d\u043a\u043e \u0424\u0440\u0438\u0434\u0440\u0438\u0445", None))
        self.action_download_img.setText(QCoreApplication.translate("MainWindow", u"\u0417\u0430\u0433\u0440\u0443\u0437\u0438\u0442\u044c", None))
        self.btn_watershed.setText(QCoreApplication.translate("MainWindow", u"Watershed", None))
        self.btn_region_growin.setText(QCoreApplication.translate("MainWindow", u"Region growing", None))
        self.btn_growing_seed.setText(QCoreApplication.translate("MainWindow", u"Growing seed", None))
        self.btn_k_means.setText(QCoreApplication.translate("MainWindow", u"K-means", None))
        self.btn_DBSCAN.setText(QCoreApplication.translate("MainWindow", u"DBSCAN", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u0412\u044b\u0431\u0440\u0430\u043d\u043d\u044b\u0439 \u0446\u0432\u0435\u0442", None))
        self.btn_drop_color.setText(QCoreApplication.translate("MainWindow", u"\u0421\u0431\u0440\u043e\u0441\u0438\u0442\u044c \u0446\u0432\u0435\u0442", None))
        self.menu.setTitle(QCoreApplication.translate("MainWindow", u"\u0424\u0430\u0439\u043b", None))
    # retranslateUi

