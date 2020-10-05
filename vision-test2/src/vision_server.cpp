
/*说明：
        该算法中，detFrameData动态数组存储了某一帧的全部目标信息，需要用.back()函数调用当前帧，假设对于当前帧的目标m，它的全部信息这么表示：
		1. ID：   detFrameData.back()[m].id;
		2. 角度： detFrameData.back()[m].objangle;
		3. 中心点： detFrameData.back()[m].objcenter;
		4. 覆盖的全向轮：  detFrameData.back()[m].coveredWheel[xx]+后续内容;  
		                       （
							       其中coveredWheel是WheelPoint类的动态数组，WheelPoint类包含了每个全向轮的信息，目标m覆盖的所有全向轮全部存储在coveredWheel这个动态数组内。
								   
								   例如： 对于目标 m 覆盖的全向轮 n ：
								          A. 全向轮编号：detFrameData.back()[m].coveredWheel[n].wID;
										  B. 全向轮中心点： detFrameData.back()[m].coveredWheel[n].wcenter；
										  C. 全向轮角度： detFrameData.back()[m].coveredWheel[n].wangle;
										  D. 目标m的中心点与该全向轮中心点的距离：detFrameData.back()[m].coveredWheel[n].wdst。
							  ）*/

#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/logger.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <vector>
#include "ros/ros.h"
#include "vision_test2/wheels_trans.h"

//#include "vision_base/vision_base_function.h"

//#define WIN "diff"

using namespace std;
using namespace cv;

enum Processor { cl, gl, cpu };

typedef struct WheelPoint
{
    double wID;
    Point2f wcenter;
    double wangle;
    double wdst;
}WheelPoint;

typedef struct TrackingBox
{
    int id;
    double objangle; //目标偏转角
    Point2f objcenter;  //目标质心点---最小外接矩形计算
    Point2f objcenter1; //上表面中心点
    Point2f objcenter2;  //下底面中心点
    vector<WheelPoint> coveredWheel;//存放当前帧被该目标覆盖的全向轮信息
}TrackingBox;

typedef struct {
    int cid;
    int cobj;
}Location;

vector<TrackingBox> object_detect(vector<Point> contour,vector<Point> contour2,vector<vector<Point>> contours,vector<TrackingBox>detData,Mat roatated2,Mat roatated22,vector<Point2f> tran_coverpoint,vector<WheelPoint>allWheel);
vector<TrackingBox> object_detect_simple(vector<Point> contour,vector<vector<Point>> contours,vector<TrackingBox>detData,Mat roatated2,vector<Point2f> tran_coverpoint,vector<WheelPoint>allWheel);
Location object_tracking(Mat roatated2,Mat roatated22,TrackingBox dbm,int objcount,vector<vector<TrackingBox>> preFrameData);



int main(int argc, char** argv)
{
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = nullptr;
    libfreenect2::PacketPipeline *pipeline = nullptr;
    ros::init(argc, argv, "wheels_pub");
    ros::NodeHandle n;
    ros::Publisher pub_1 = n.advertise<vision_test2::wheels_trans>("wheel_number", 1);
    vision_test2::wheels_trans wheels;
    if(freenect2.enumerateDevices() == 0)
    {
        std::cout << "no device connected!" << std::endl;
        return -1;
    }
    string serial = freenect2.getDefaultDeviceSerialNumber();
    if(serial == "")  return -1;
    cout<<"The serial number is :"<<serial<<endl;

    int depthProcessor = Processor::cl;
    if(depthProcessor == Processor::cpu)
    {
        if(!pipeline)
        {
            pipeline = new libfreenect2::CpuPacketPipeline();
        }
    }
    else
    if (depthProcessor == Processor::gl)
    {
#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
        if(!pipeline)
            pipeline = new libfreenect2::OpenGLPacketPipeline();
#else
        std::cout << "OpenGL pipeline is not supported!" << std::endl;
#endif
    }
    else
    if (depthProcessor == Processor::cl)
    {
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
        if(!pipeline)
            pipeline = new libfreenect2::OpenCLPacketPipeline();
#else
        std::cout << "OpenCL pipeline is not supported!" << std::endl;
#endif
    }

    if(pipeline)
    {
        dev = freenect2.openDevice(serial, pipeline);
    }
    else
    {
        dev = freenect2.openDevice(serial);
    }
    if(dev == 0)
    {
        std::cout << "failure opening device!" << std::endl;
        return -1;
    }

    //! [listeners]
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Depth);
    libfreenect2::FrameMap frames;
    dev->setIrAndDepthFrameListener(&listener);

    dev->start();
    std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
    std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;
    //! [start]
    Mat depthmat, depthmatUndistorted, rgbd, rgbd2;

    //------------读取平台所有模块上全向轮的编号、中心点和角度信息并存入数组--------------
    ifstream in("/home/zhaojiaqi/vision-module/src/vision_module/wheels_num.txt");
    istringstream iss;
    string line;
    double temp;
    vector<WheelPoint>allWheel;
    vector<Point2f> coverpoint;
    if (in)
    {
        while (getline(in, line))
        {
            WheelPoint wheel;
            int i = 0;

            iss.clear();
            iss.str(line);
            while (iss >> temp)
            {
                //cout << t << " ";
                if (i == 0)
                {
                    wheel.wID = temp; i++; continue;
                }
                if (i == 1)
                {
                    wheel.wcenter.x = temp; i++; continue;
                }
                if (i == 2)
                {
                    wheel.wcenter.y = temp; i++; continue;
                }
                if (i == 3)
                {
                    wheel.wangle = temp;
                }
            }
            allWheel.push_back(wheel);
            coverpoint.push_back(wheel.wcenter);
        }
    }
    in.close();

    //----------------视觉检测部分---------------------

    //-----跟踪参数-----
    vector<TrackingBox> detData;    //当前帧各个目标的信息
    vector<vector<TrackingBox>> detFrameData;   //当前帧所有目标
    vector<vector<TrackingBox>> preFrameData;   //上一帧所有目标

    vector<WheelPoint> coveredWheel;  //存放当前帧被目标覆盖的全向轮信息

    int objcount = 0; //定义出现过的目标总数

    //*************根据实际情况修改**************

	//此处输入平台尺寸，单位毫米
	Point2f srcvert[4];
	srcvert[0] = Point(0, 0);//左上角
	srcvert[1] = Point(1600, 0);//右上角
	srcvert[2] = Point(1600, 1870);//右下角
	srcvert[3] = Point(0, 1870);//左下角

	//此处输入在原图像中平台四个顶点的坐标，为像素坐标
        Point2f dstvert[4];
        dstvert[0] = Point(104,14);
        dstvert[1] = Point(354, 14);
        dstvert[2] = Point(354, 294);
        dstvert[3] = Point(104, 294);

        //此处为变换后的平台边长，为了去除可能的倾斜畸变以及转换全向轮坐标。将平台尺寸1600*1450等比例缩小到kinect2的分辨率512*424范围内，且与原图像中平台像素尺寸接近
        Point2f  dstvert2[4];
        dstvert2[0] = Point(0,0);
        dstvert2[1] = Point(250, 0);
        dstvert2[2] = Point(250, 280);
        dstvert2[3] = Point(0, 280);

        //******************************************

        vector<Point2f> tran_coverpoint;
        Mat warp = getPerspectiveTransform(srcvert, dstvert2);
        perspectiveTransform(coverpoint,tran_coverpoint,warp);


        Mat warp2 = getPerspectiveTransform(dstvert, dstvert2);




    for (int p = 0;; p++)
    {
        listener.waitForNewFrame(frames);
        libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];
        //! [loop start]
        Mat depthmat =cv::Mat(depth->height, depth->width, CV_32FC1, depth->data);
        Mat a=depthmat/4500.0f;
        
        a.convertTo(a,CV_8UC1,255.0);
	Mat dep;
	depthmat.copyTo(dep);

        imshow("srcvideo",a);//相机原图像


        
        Mat roatated,roatated22;
        warpPerspective(dep, roatated, warp2, Size(250,280), INTER_LINEAR, BORDER_CONSTANT);

	Mat roatated2 = roatated.clone();

	const int channels = roatated2.channels();
	int nRows = roatated2.rows;
	int nCols = roatated2.cols*channels;

	float *ptn;
	for (int i = 0; i<roatated2.rows; i++)
	{
		ptn = roatated2.ptr<float>(i);//获取每行首地址
		for (int j = 0; j<roatated2.cols; j++)
		{
			if (ptn[j]<50)
				ptn[j] = 915;
		}
	}

		
        imshow("depth", roatated2/4500.0f);
	roatated2.copyTo(roatated22);
	
	
	threshold(roatated2, roatated2, 890, 4500, 1);
	threshold(roatated22, roatated22, 790, 4500, 1);

	roatated2.convertTo(roatated2, CV_8UC1, 255.0);
        roatated22.convertTo(roatated22, CV_8UC1, 255.0);

	medianBlur(roatated2, roatated2, 5);
	medianBlur(roatated22, roatated22, 5);

        vector<vector<Point>>contours;
	vector<vector<Point>>contours2;

        findContours(roatated2, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	findContours(roatated22, contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

       for (int i = 0; i < contours.size(); i++)
	   {
		   double area = contourArea(contours[i]);
		   if (area > 350) {
			   
			   //椭圆检测
			   if (contours[i].size() > 5)
			   {
				if(contours.size()==contours2.size())
				   {
					detData=object_detect(contours[i],contours2[i],contours,detData,roatated2,roatated22,tran_coverpoint,allWheel);
				   }
				else
				   {
					detData=object_detect_simple(contours[i],contours,detData,roatated2,tran_coverpoint,allWheel);
                                   }
			   }
		   }
        }

        //************************************跟踪算法*********************************************
       if(detData.size()>0)
       {
           
	   detFrameData.push_back(detData);//**********存储当前帧目标信息，当前帧目标的全部信息全部存储在detFrameData队列中*******
           
           detData.clear();//删除当前帧目标队列，下一帧再继续存储
           //cout << detFrameData.size() << endl;

	   //*********存储目标ID**********
           if (detFrameData.size() >= 1)//当有目标检测到之后再运行循环
           {
               if (detFrameData.size() == 1)//当第一次出现目标时，直接赋予ID
               {
                   for (int i = 0; i < detFrameData[0].size(); i++)
                   {
                       detFrameData[0][i].id = i;
                   }
                   objcount = detFrameData[0].size();// 目标总数存储为第一帧所有的目标数
                   preFrameData.push_back(detFrameData.back());//第一帧的所有目标存储为上一帧的目标
                   //continue;
               }

               //--------------主循环，当前帧的所有目标分别与上一帧的所有目标进行IOU或中心点欧式距离计算，匹配到的继承上一帧ID，未匹配到的判定为新目标并赋予新ID---------------------------

               if (detFrameData.size() > 1)
               {
                   for (int m = 0; m < detFrameData.back().size(); m++) {
                        Location c_tb;
                        c_tb=object_tracking(roatated2,roatated22,detFrameData.back()[m],objcount,preFrameData);
                        detFrameData.back()[m].id=c_tb.cid;
                        objcount=c_tb.cobj;
                    }


                    //先测试单物体的，后面再测试多物体的。
                    //下面把收到的数据都发出来、
                    vector<int> tmp_1;
                    // cout<< detFrameData.back()[0].coveredWheel.size()<<endl;
                    if (detFrameData.back()[0].coveredWheel.size() > 0)
                    {
                        for(int j = 0;j < detFrameData.back()[0].coveredWheel.size();j++)
                        {
                            tmp_1.push_back(detFrameData.back()[0].coveredWheel[j].wID);
                        }    
                        wheels.wID = tmp_1;
                        wheels.wcenterx = detFrameData.back()[0].objcenter.x;
                        wheels.wcentery = detFrameData.back()[0].objcenter.y;
                        wheels.angle = detFrameData.back()[0].objangle;
                        pub_1.publish(wheels);
                        ros::spinOnce();
                    }


                    // for(int j = 0;j < detFrameData.back()[0].coveredWheel.size();j++)
                    // {
                    //     tmp_1.push_back(detFrameData.back()[0].coveredWheel[j].wID);
                    // }    
                    // wheels.wID = tmp_1;
                    // wheels.wcenterx = detFrameData.back()[0].objcenter.x;
                    // wheels.wcentery = detFrameData.back()[0].objcenter.y;
                    // wheels.angle = detFrameData.back()[0].objangle;

                   preFrameData.clear();  //清空上一帧全部目标
                   preFrameData.push_back(detFrameData.back());  //当前帧的全部目标成为新的上一帧全部目标
		  
		  
                   detFrameData.pop_back(); //删除当前帧的目标信息，准备迎接下一帧，防止因存储帧数过多导致内存不足
               }
           }
       }

        imshow("frame",roatated2);
	imshow("frame2",roatated22);

        listener.release(frames);
               //这里我们先设置一个物体，后面我们再设置多个物体的情况。


        if (waitKey(1) == 27)
            break;
        // if (key == '\x03')
        // {
        //     printf("\n");
        //     break;
        // }
    }

    dev->stop();
    dev->close();

    return 0;
}


vector<TrackingBox> object_detect(vector<Point> contour,vector<Point> contour2,vector<vector<Point>> contours,vector<TrackingBox>detData,Mat roatated2,Mat roatated22,vector<Point2f> tran_coverpoint,vector<WheelPoint>allWheel)
{
    double covered = 2;
    RotatedRect box = fitEllipse(contour);

    //最小外接矩形
    RotatedRect minbox = minAreaRect(contour);
    RotatedRect minbox2 = minAreaRect(contour2);

    Point2f vertex[4];
    minbox.points(vertex);
    drawContours(roatated2, contours, -1, Scalar(0, 0, 0), 1, 8, vector<Vec4i>());
    circle(roatated2, minbox.center, 5, Scalar(0, 0, 0), -1, 8, 0);//质心位置
    circle(roatated22, minbox2.center, 5, Scalar(0, 0, 0), -1, 8, 0);//质心位置

    //角度全部转换到-90~90之间
    double frameangle;
    if (box.angle > 0 && box.angle < 270.0 || box.angle == 270.0)
        frameangle = 90 - box.angle;
    else if (box.angle > 270.0 && box.angle < 360.0 || box.angle == 360.0)
        frameangle = box.angle - 360.0;


    TrackingBox tb;
    tb.objangle = frameangle;//******存储目标角度******
    //tb.objcenter = minbox.center;//*******存储目标中心点******
    tb.objcenter1 = minbox.center;
    tb.objcenter2=minbox2.center;
    tb.objcenter=Point2f (minbox.center.x+(minbox.center.x-minbox2.center.x),minbox.center.y+(minbox.center.y-minbox2.center.y));


    //*********判断哪些全向轮被覆盖，并存储***********
    for (int m = 0; m < allWheel.size(); m++)
    {
        covered = pointPolygonTest(contour, tran_coverpoint[m], false);
        if (covered != -1)
        {
            tb.coveredWheel.push_back(allWheel[m]);
            tb.coveredWheel.back().wdst = sqrt(pow(allWheel[m].wcenter.x - tb.objcenter.x, 2) + pow(allWheel[m].wcenter.y - tb.objcenter.y, 2));
        }
    }
    detData.push_back(tb);
    return detData;
}


vector<TrackingBox> object_detect_simple(vector<Point> contour,vector<vector<Point>> contours,vector<TrackingBox>detData,Mat roatated2,vector<Point2f> tran_coverpoint,vector<WheelPoint>allWheel)
{
    double covered = 2;
    RotatedRect box = fitEllipse(contour);

    //最小外接矩形
    RotatedRect minbox = minAreaRect(contour);

    Point2f vertex[4];
    minbox.points(vertex);
    drawContours(roatated2, contours, -1, Scalar(0, 0, 0), 1, 8, vector<Vec4i>());
    circle(roatated2, minbox.center, 5, Scalar(0, 0, 0), -1, 8, 0);//质心位置

    //角度全部转换到-90~90之间
    double frameangle;
    if (box.angle > 0 && box.angle < 270.0 || box.angle == 270.0)
        frameangle = 90 - box.angle;
    else if (box.angle > 270.0 && box.angle < 360.0 || box.angle == 360.0)
        frameangle = box.angle - 360.0;


    TrackingBox tb;
    tb.objangle = frameangle;//******存储目标角度******
    tb.objcenter = minbox.center;//*******存储目标中心点******


    //*********判断哪些全向轮被覆盖，并存储***********
    for (int m = 0; m < allWheel.size(); m++)
    {
        covered = pointPolygonTest(contour, tran_coverpoint[m], false);
        if (covered != -1)
        {
            tb.coveredWheel.push_back(allWheel[m]);
            tb.coveredWheel.back().wdst = sqrt(pow(allWheel[m].wcenter.x - tb.objcenter.x, 2) + pow(allWheel[m].wcenter.y - tb.objcenter.y, 2));
        }
    }
    detData.push_back(tb);
    return detData;
}


Location object_tracking(Mat roatated2,Mat roatated22,TrackingBox dbm,int objcount,vector<vector<TrackingBox>> preFrameData)
{
        Location tb;
        bool matched = false;
        for (int n = 0; n < preFrameData.back().size(); n++)
        {
            double dist = sqrt(
                    pow(dbm.objcenter.x - preFrameData.back()[n].objcenter.x, 2) +
                    pow(dbm.objcenter.y - preFrameData.back()[n].objcenter.y, 2));
            if (dist < 50)
            {
                dbm.id = preFrameData.back()[n].id;
                matched = true;
                preFrameData.back().erase(preFrameData.back().begin() + n);
                continue;
            }
        }
        if (matched == false)
        {
            dbm.id = objcount;
            objcount += 1;
        }
        char tam[100];
        sprintf(tam, "%d ", dbm.id);
        putText(roatated2, tam, dbm.objcenter, FONT_HERSHEY_SIMPLEX, 0.8,
                Scalar(0,0,0),
                1);
        tb.cid=dbm.id;
        tb.cobj=objcount;
        return tb;
}


