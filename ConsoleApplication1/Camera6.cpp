#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <time.h>
#include <sstream>

#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace cv;

static int framecounter=0;
Mat frame1, frame2,current_frame,prev_frame,next_frame;
Mat src, src_gray;
Mat dst, detected_edges;
Mat d1, d2, motion,refree;
int edgeThresh = 1;
int lowThreshold=30;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";
double area=0.0;
int reference=24000;
int difference=5000;
int number_of_changes=0, number_of_sequence = 0;
/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void*)
{
	if(framecounter%3==0)
	{
  /// Reduce noise with a kernel 3x3
  blur( src_gray, detected_edges, Size(3,3) );

  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);

  frame2.copyTo( dst, detected_edges);
    cvtColor( dst, dst, CV_RGB2GRAY );
  threshold(dst, dst, 0, 255, CV_THRESH_BINARY);
  int WhitePixels = countNonZero(dst);
	if(lowThreshold >20)
		  WhitePixels-=difference;

  area=((double)WhitePixels/(double)reference)*100;

	if(area>100)
		  area=100;
	cout<<"AREA:"<<area<<"%\t";
  imshow( window_name, dst );
	}
}



// Check if there is motion in the result matrix
// count the number of changes and return.
inline int detectMotion(const Mat & motion, Mat & result, Mat & result_cropped,
                 int x_start, int x_stop, int y_start, int y_stop,
                 int max_deviation,
                 Scalar & color)
{
    // calculate the standard deviation
    Scalar mean, stddev;
    meanStdDev(motion, mean, stddev);
    // if not to much changes then the motion is real (neglect agressive snow, temporary sunlight)
    if(stddev[0] < max_deviation)
    {
        int number_of_changes = 0;
        int min_x = motion.cols, max_x = 0;
        int min_y = motion.rows, max_y = 0;
        // loop over image and detect changes
        for(int j = y_start; j < y_stop; j+=2){ // height
            for(int i = x_start; i < x_stop; i+=2){ // width
                // check if at pixel (j,i) intensity is equal to 255
                // this means that the pixel is different in the sequence
                // of images (prev_frame, current_frame, next_frame)
                if(static_cast<int>(motion.at<uchar>(j,i)) == 255)
                {
                    number_of_changes++;
                    if(min_x>i) min_x = i;
                    if(max_x<i) max_x = i;
                    if(min_y>j) min_y = j;
                    if(max_y<j) max_y = j;
                }
            }
        }
        if(number_of_changes)
		{
            //check if not out of bounds
            if(min_x-10 > 0) min_x -= 10;
            if(min_y-10 > 0) min_y -= 10;
            if(max_x+10 < result.cols-1) max_x += 10;
            if(max_y+10 < result.rows-1) max_y += 10;
            // draw rectangle round the changed pixel
            Point x(min_x,min_y);
            Point y(max_x,max_y);
            Rect rect(x,y);
            Mat cropped = result(rect);
            cropped.copyTo(result_cropped);
            rectangle(result,rect,color,1);
        }
        return number_of_changes;
    }
    return 0;
}


static void onMouse( int event, int x, int y, int f, void* )
{
    cout << x << " " << y << endl;
    //putText(image, "point", Point(x,y), CV_FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0));
}

void rotate(cv::Mat& src, double angle, cv::Mat& dst)
{
    int len =std::max(src.cols, src.rows);
    cv::Point2f pt(len/2., len/2.);
    cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);

    cv::warpAffine(src, dst, r, cv::Size(len, len));
}
	   int maximum(int x, int y, int z) {
	int max = x; /* assume x is the largest */

	if (y > max)
	{ /* if y is larger than max, assign y to max */
		max = y;
	} /* end if */

	if (z > max) 
	{ /* if z is larger than max, assign z to max */
		max = z;
	} /* end if */

	return max; /* max is the largest value */
}
int main (int argc, char * const argv[])
{
	int t1=0,t2=0,t3=0,t4=0,t5=0;
	int counttraffic=0;
		int numberofframes=7; int i=0;
		float k=(float)2/8;
		float k1;
		k1=(float)1.0-k;
	vector<int> compression_params;
    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);
	float average=0.0;
	int low=0,med=0,high=0;
	Mat tmp;
   

    Mat result, result_cropped;
	cv::Rect myROI(147, 22, 492, 456);
	//Change threshold value for area based on the system time(and hence the actual time)
// assumption: system time shows current IST time
/*	
	if(h>7 && h<=9)
		lowThreshold=25;
	else if(h>9 && h<=16)
		lowThreshold=30;
	else if(h>16 && h<=18)
		lowThreshold=25;
	else if(h>18 && h<=19)
		lowThreshold=18;
	else if(h>19 && h <=23)
		lowThreshold=15;
	else if(h>=0 && h<=7)
		lowThreshold=15;

*/


	int stop=1;
    // Take images and convert them to gray
	char* videoFilename =argv[1];
	 VideoCapture capture(videoFilename);
	 int chk=0;
    if(!capture.isOpened())
	{
        //error in opening the video input
        cerr << "Unable to open video file: " << videoFilename << endl;
        exit(EXIT_FAILURE);
    }

   int key=0;


Mat croppedImage;
    while( (char)key != 'q' && (char)key != 27 )
	{

	 if(!capture.read(frame2)) 
	 {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
     }
	 		// Setup a rectangle to define your region of interest

		imshow("orginal",frame2);
		//    setMouseCallback("org", onMouse, 0 );
		if(chk==0)
		{
			frame1=frame2.clone();
			chk=10;
			// Crop the full image to that image contained by the rectangle myROI
			// Note that this doesn't copy the data
			croppedImage = frame1(myROI);
			rectangle( croppedImage, Point( 0, 0 ), Point( 70, 80), Scalar( 0, 0, 0 ), -1, 8 );
			if(croppedImage.empty())
			cout<<"empty";
			cvtColor(croppedImage, prev_frame, CV_RGB2GRAY);
		}
		frame2 = frame2(myROI);
		
	rectangle( frame2, Point( 0, 0 ), Point( 70, 80), Scalar( 0, 0, 0 ), -1, 8 );
	cv::imshow("stream",frame2);
	if(frame2.empty())
			cout<<"empty";
 
    cvtColor(frame2, current_frame, CV_RGB2GRAY);
	cvtColor(frame2, src_gray, CV_RGB2GRAY);
 
   /// Create a matrix of the same type and size as src (for dst)
	dst.create( frame2.size(), frame2.type() );


  
  /// Create a window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Create a Trackbar for user to enter threshold
 // createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );

  /// Show the image
  CannyThreshold(15, 0);

 if(next_frame.empty()==true)
	next_frame=prev_frame.clone();
    // d1 and d2 for calculating the differences
    // result, the result of and operation, calculated on d1 and d2
    // number_of_changes, the amount of changes in the result matrix.
    // color, the color for drawing the rectangle when something has changed.
    Scalar mean_, color(0,255,255); // yellow
    
    // Detect motion in window
    int x_start = 0, x_stop = current_frame.cols;
	int y_start = 0, y_stop = current_frame.rows;

    // If more than 'there_is_motion' pixels are changed, we say there is motion
    // and store an image on disk
    int there_is_motion = 3;
    
    // Maximum deviation of the image, the higher the value, the more motion is allowed
    int max_deviation = 1000;
    
    // Erode kernel
    Mat kernel_ero = getStructuringElement(MORPH_RECT, Size(2,2));
    
    // All settings have been set, now go in endless loop and
    // take as many pictures you want..
       // Take a new image
	
	if(framecounter%3==0)
	{
		result = current_frame.clone();
	//	cv::imshow("next",next_frame);
	//	cv::imshow("prev",prev_frame);
	//	cv::imshow("current",current_frame);
        // Calc differences between the images and do AND-operation
        // threshold image, low differences are ignored (ex. contrast change due to sunlight)
		absdiff(next_frame, prev_frame, d1);
		//imshow("d1",d1);
		absdiff(prev_frame, current_frame, d2);
        //imshow("d2",d2);
		bitwise_and(d1, d2, motion);
	//	imshow("motion",motion);
        threshold(motion, motion, 35, 255, CV_THRESH_BINARY);
        erode(motion, motion, kernel_ero);

		//number_of_changes = detectMotion(area, result, result_cropped,  x_start, x_stop, y_start, y_stop, max_deviation, color);
        number_of_changes = detectMotion(motion, result, result_cropped,  x_start, x_stop, y_start, y_stop, max_deviation, color);
	
        // If a lot of changes happened, we assume something changed.
		
		if(i<=numberofframes)
		{
			t5=number_of_changes;
			t4=t5;
			t3=t4;
			t2=t3;
			t1=t2;
			average = (float)(t1+t2+t3+t4+t5)/5;
		}
		if(i>numberofframes) 
			average= number_of_changes * k + average * k1;
			//cout<<"number: "<<number_of_changes<<endl;
		if(average <1 && area > 40)
			cout<<"Traffic : STAGNANT!!!\t";
		else if(average >= 10 && average < 800 && (area < 40))
			cout<<"Traffic : LOW\t";
		else if(average >=800 && average < 2000 || (area >=40 && area <85))
			cout<<"Traffic : MEDIUM\t";
		else if(average <=100 &&  area >=85 )
			cout<<"Traffic : HIGH\t";
		cout<<"Pixel Change:"<<average<<endl;
		next_frame = prev_frame.clone();
        prev_frame = current_frame.clone();
		if(framecounter > 5000)
			framecounter=0;
	}	
        
	
		//	cout<<framecounter<<endl;
		framecounter++;

        cvWaitKey (30);
		i++;
  }
	return 0;
}


