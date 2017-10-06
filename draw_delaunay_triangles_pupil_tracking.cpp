#include <stdio.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>
#include <fstream>
#include <queue>
#include <cmath>


using namespace cv;
using namespace std;

//JUSTIN CHANGE THE PATH TO THE haarcascade_frontalface_alt.xml FILE
String face_cascade_name = "/Users/shelleywu/Desktop/everything/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;


//debugging
const bool kPlotVectorField = false;
// Size constants
const int kEyePercentTop = 25;
const int kEyePercentSide = 13;
const int kEyePercentHeight = 30;
const int kEyePercentWidth = 35;

// Preprocessing
const bool kSmoothFaceImage = false;
const float kSmoothFaceFactor = 0.005;

// Algorithm Parameters
const int kFastEyeWidth = 50;
const int kWeightBlurSize = 5;
const bool kEnableWeight = true;
const float kWeightDivisor = 1.0;
const double kGradientThreshold = 50.0;

// Postprocessing
const bool kEnablePostProcess = true;
const float kPostProcessThreshold = 0.97;

Point2f leftPupil;
Point2f rightPupil;

//Point findEyeCenter methods
// Pre-declarations
Mat floodKillEdges(Mat &mat);

void findEyes(Mat frame_gray, Rect face);

typedef struct face_landmark_node 
{
  int frame;
  int indice;
  float x;
  float y;
  struct face_landmark_node *next;
} face_landmark_node;

typedef struct pixel_location_node
{
    int pixel_index;
    float pixel_location_x;
    float pixel_location_y;
    struct pixel_location_node *next;
} pixel_location_node;


pixel_location_node *pixel_head = NULL;
pixel_location_node *pixel_current = NULL;

int pln_index = 0; //pixel_location_node index counter

int line_number = 1;

//left eyeball
float le_left_top[2]; //41
float le_right_bottom[2]; //44
float le_left_bottom[2]; //45
float le_right_top[2]; //42

//right eyeball
float re_right_top[2]; //48
float re_left_bottom[2]; //51
float re_left_top[2]; //47
float re_right_bottom[2]; //50



static void draw_point (Mat &img, Point2f fp, Scalar color)
{
  circle (img, fp, 3, color, CV_FILLED, CV_AA, 0);
}

//helper functions
bool rectInImage(Rect rect, Mat image) {
    return rect.x > 0 && rect.y > 0 && rect.x+rect.width < image.cols &&
    rect.y+rect.height < image.rows;
}

bool inMat(Point p,int rows,int cols) {
    return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

Mat matrixMagnitude(const Mat &matX, const Mat &matY) {
    Mat mags(matX.rows,matX.cols,CV_64F);
    for (int y = 0; y < matX.rows; ++y) {
        const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
        double *Mr = mags.ptr<double>(y);
        for (int x = 0; x < matX.cols; ++x) {
            double gX = Xr[x], gY = Yr[x];
            double magnitude = sqrt((gX * gX) + (gY * gY));
            Mr[x] = magnitude;
        }
    }
    return mags;
}

double computeDynamicThreshold(const Mat &mat, double stdDevFactor) {
    Scalar stdMagnGrad, meanMagnGrad;
    meanStdDev(mat, meanMagnGrad, stdMagnGrad);
    double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
    return stdDevFactor * stdDev + meanMagnGrad[0];
}

#pragma mark Visualization
/*
 template<typename T> mglData *matToData(const cv::Mat &mat) {
 mglData *data = new mglData(mat.cols,mat.rows);
 for (int y = 0; y < mat.rows; ++y) {
 const T *Mr = mat.ptr<T>(y);
 for (int x = 0; x < mat.cols; ++x) {
 data->Put(((mreal)Mr[x]),x,y);
 }
 }
 return data;
 }
 
 void plotVecField(const cv::Mat &gradientX, const cv::Mat &gradientY, const cv::Mat &img) {
 mglData *xData = matToData<double>(gradientX);
 mglData *yData = matToData<double>(gradientY);
 mglData *imgData = matToData<float>(img);
 
 mglGraph gr(0,gradientX.cols * 20, gradientY.rows * 20);
 gr.Vect(*xData, *yData);
 gr.Mesh(*imgData);
 gr.WriteFrame("vecField.png");
 
 delete xData;
 delete yData;
 delete imgData;
 }*/

#pragma mark Helpers


Mat computeMatXGradient(const Mat &mat) {
    Mat out(mat.rows,mat.cols,CV_64F);
    
    for (int y = 0; y < mat.rows; ++y) {
        const uchar *Mr = mat.ptr<uchar>(y);
        double *Or = out.ptr<double>(y);
        
        Or[0] = Mr[1] - Mr[0];
        for (int x = 1; x < mat.cols - 1; ++x) {
            Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
        }
        Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
    }
    
    return out;
}

#pragma mark Main Algorithm

void testPossibleCentersFormula(int x, int y, const Mat &weight,double gx, double gy, Mat &out) {
    // for all possible centers
    for (int cy = 0; cy < out.rows; ++cy) {
        double *Or = out.ptr<double>(cy);
        const unsigned char *Wr = weight.ptr<unsigned char>(cy);
        for (int cx = 0; cx < out.cols; ++cx) {
            if (x == cx && y == cy) {
                continue;
            }
            // create a vector from the possible center to the gradient origin
            double dx = x - cx;
            double dy = y - cy;
            // normalize d
            double magnitude = sqrt((dx * dx) + (dy * dy));
            dx = dx / magnitude;
            dy = dy / magnitude;
            double dotProduct = dx*gx + dy*gy;
            dotProduct = std::max(0.0,dotProduct);
            // square and multiply by the weight
            if (kEnableWeight) {
                Or[cx] += dotProduct * dotProduct * (Wr[cx]/kWeightDivisor);
            } else {
                Or[cx] += dotProduct * dotProduct;
            }
        }
    }
}

Point findEyeCenter(Mat face, Rect eye) {
    Mat eyeROIUnscaled = face(eye);
    //Mat eyeROI;
    //scaleToFastSize(eyeROIUnscaled, eyeROI);
    // draw eye region
    //rectangle(face,eye,1234);
    //-- Find the gradient
    Mat gradientX = computeMatXGradient(eyeROIUnscaled);
    Mat gradientY = computeMatXGradient(eyeROIUnscaled.t()).t();
    //-- Normalize and threshold the gradient
    // compute all the magnitudes
    Mat mags = matrixMagnitude(gradientX, gradientY);
    //compute the threshold
    double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);
    //double gradientThresh = kGradientThreshold;
    //double gradientThresh = 0;
    //normalize
    for (int y = 0; y < eyeROIUnscaled.rows; ++y) {
        double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
        const double *Mr = mags.ptr<double>(y);
        for (int x = 0; x < eyeROIUnscaled.cols; ++x) {
            double gX = Xr[x], gY = Yr[x];
            double magnitude = Mr[x];
            if (magnitude > gradientThresh) {
                Xr[x] = gX/magnitude;
                Yr[x] = gY/magnitude;
            } else {
                Xr[x] = 0.0;
                Yr[x] = 0.0;
            }
        }
    }
    //imshow(debugWindow,gradientX);
    //-- Create a blurred and inverted image for weighting
    Mat weight;
    GaussianBlur( eyeROIUnscaled, weight, Size( kWeightBlurSize, kWeightBlurSize ), 0, 0 );
    for (int y = 0; y < weight.rows; ++y) {
        unsigned char *row = weight.ptr<unsigned char>(y);
        for (int x = 0; x < weight.cols; ++x) {
            row[x] = (255 - row[x]);
        }
    }
    //imshow(debugWindow,weight);
    //-- Run the algorithm!
    Mat outSum = Mat::zeros(eyeROIUnscaled.rows,eyeROIUnscaled.cols,CV_64F);
    // for each possible gradient location
    // Note: these loops are reversed from the way the paper does them
    // it evaluates every possible center for each gradient location instead of
    // every possible gradient location for every center.
    //printf("Eye Size: %ix%i\n",outSum.cols,outSum.rows);
    for (int y = 0; y < weight.rows; ++y) {
        const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
        for (int x = 0; x < weight.cols; ++x) {
            double gX = Xr[x], gY = Yr[x];
            if (gX == 0.0 && gY == 0.0) {
                continue;
            }
            testPossibleCentersFormula(x, y, weight, gX, gY, outSum);
        }
    }
    // scale all the values down, basically averaging them
    double numGradients = (weight.rows*weight.cols);
    Mat out;
    outSum.convertTo(out, CV_32F,1.0/numGradients);
    //imshow(debugWindow,out);
    //-- Find the maximum point
    Point maxP;
    double maxVal;
    minMaxLoc(out, NULL,&maxVal,NULL,&maxP);
    //printf("Before kEnablePostProcess condition: %d %d %f \n", maxP.x, maxP.y, maxVal);
    //-- Flood fill the edges
    if(kEnablePostProcess) {
        //printf("Passes kEnablePostProcess condition \n");
        Mat floodClone;
        //double floodThresh = computeDynamicThreshold(out, 1.5);
        double floodThresh = maxVal * kPostProcessThreshold;
        //printf("floodThresh: %f \n", floodThresh);
        threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);
        if(kPlotVectorField) {
            //printf("passes kPlotVectorField \n");
            //plotVecField(gradientX, gradientY, floodClone);
            imwrite("eyeFrame.png",eyeROIUnscaled);
        }
        Mat mask = floodKillEdges(floodClone);
        //imshow(debugWindow + " Mask",mask);
        //imshow(debugWindow,out);
        // redo max
        minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);
    }
    
    //printf("maxP: %d %d \n", maxP.x, maxP.y);
    
    return maxP;
}

#pragma mark Postprocessing

bool floodShouldPushPoint(const Point &np, const Mat &mat) {
    return inMat(np, mat.rows, mat.cols);
}

// returns a mask
Mat floodKillEdges(Mat &mat) {
    rectangle(mat,Rect(0,0,mat.cols,mat.rows),255);
    
    Mat mask(mat.rows, mat.cols, CV_8U, 255);
    queue<Point> toDo;
    toDo.push(Point(0,0));
    while (!toDo.empty()) {
        Point p = toDo.front();
        toDo.pop();
        if (mat.at<float>(p) == 0.0f) {
            continue;
        }
        // add in every direction
        Point np(p.x + 1, p.y); // right
        if (floodShouldPushPoint(np, mat)) toDo.push(np);
        np.x = p.x - 1; np.y = p.y; // left
        if (floodShouldPushPoint(np, mat)) toDo.push(np);
        np.x = p.x; np.y = p.y + 1; // down
        if (floodShouldPushPoint(np, mat)) toDo.push(np);
        np.x = p.x; np.y = p.y - 1; // up
        if (floodShouldPushPoint(np, mat)) toDo.push(np);
        // kill it
        mat.at<float>(p) = 0.0f;
        mask.at<uchar>(p) = 0;
    }
    return mask;
}

void findEyes(Mat frame_gray, Rect face)
{
    Mat faceROI = frame_gray(face);
    
    float eye_region_width = face.width * (kEyePercentWidth/100.0);
    float eye_region_height = face.width * (kEyePercentHeight/100.0);
    float eye_region_top = face.height * (kEyePercentTop/100.0);
    
    Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),
                       eye_region_top,eye_region_width,eye_region_height);
    Rect rightEyeRegion(face.width - eye_region_width - face.width*(kEyePercentSide/100.0),
                        eye_region_top,eye_region_width,eye_region_height);
    
    
    leftPupil = findEyeCenter(faceROI,leftEyeRegion);
    rightPupil = findEyeCenter(faceROI,rightEyeRegion);
    
    // get corner regions
    Rect leftRightCornerRegion(leftEyeRegion);
    leftRightCornerRegion.width -= leftPupil.x;
    leftRightCornerRegion.x += leftPupil.x;
    leftRightCornerRegion.height /= 2;
    leftRightCornerRegion.y += leftRightCornerRegion.height / 2;
    Rect leftLeftCornerRegion(leftEyeRegion);
    leftLeftCornerRegion.width = leftPupil.x;
    leftLeftCornerRegion.height /= 2;
    leftLeftCornerRegion.y += leftLeftCornerRegion.height / 2;
    Rect rightLeftCornerRegion(rightEyeRegion);
    rightLeftCornerRegion.width = rightPupil.x;
    rightLeftCornerRegion.height /= 2;
    rightLeftCornerRegion.y += rightLeftCornerRegion.height / 2;
    Rect rightRightCornerRegion(rightEyeRegion);
    rightRightCornerRegion.width -= rightPupil.x;
    rightRightCornerRegion.x += rightPupil.x;
    rightRightCornerRegion.height /= 2;
    rightRightCornerRegion.y += rightRightCornerRegion.height / 2;
    // change eye centers to face coordinates
    rightPupil.x += rightEyeRegion.x;
    rightPupil.y += rightEyeRegion.y;
    leftPupil.x += leftEyeRegion.x;
    leftPupil.y += leftEyeRegion.y;
}

void detectEyes(Mat frame)
{
    vector<Rect> faces;
    
    vector<Mat> rgbChannels(3);
    split(frame, rgbChannels);
    Mat frame_gray = rgbChannels[2];
    
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, Size(150, 150) );
    
    //printf("faces[i]: %d %d \n", faces[0].x, faces[0].y);
    
    
    if (faces.size() > 0) {
        findEyes(frame_gray, faces[0]);
    }
    
    leftPupil.x += faces[0].x;
    leftPupil.y += faces[0].y;
    
    rightPupil.x += faces[0].x;
    rightPupil.y += faces[0].y;
    //printf("line 400 \n");
}
 
static void draw_delaunay (Mat &img, Subdiv2D &subdiv, Scalar delaunay_color)
{
  vector<Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);
    //printf("line 402 passed \n");
  vector<Point> pt(3);
  Size size = img.size();
  Rect rect(0,0, size.width, size.height);
 
  for (size_t i = 0; i < triangleList.size(); i++)
  {
    Vec6f t = triangleList[i];
    pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
    pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
    pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
         
    // Draw rectangles completely inside the image.
    if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
    {
      line (img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
      line (img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
      line (img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
    }
  }
}




static void run (face_landmark_node *face_landmark_list_head, char *file)
{
  face_landmark_node *face_landmark_element;
  Scalar delaunay_color(255,0,0), points_color(0, 0, 255), pupils_color(0, 255,0); // Note: delaunay_color and points_color are in BGR (BLUE, GREEN, RED) format
  Mat source_image;
  Size source_image_resolution;
  char input_filename[1280], output_filename[1280]; // 1024 bytes for path + 256 bytes for filename  = 1280 bytes.

  memset (&input_filename, 0, sizeof(input_filename) - 1);
  memset (&output_filename, 0, sizeof(input_filename) - 1);
  strncpy (&input_filename[0], file, sizeof(input_filename) - 1);
  snprintf (&output_filename[0], sizeof(output_filename) - 1,  "OUTPUT-%s", &input_filename[0]);
    //printf("about to run this image \n");
  if (input_filename[0] != '\0')
  {
    source_image = imread (&input_filename[0]);
    if (!source_image.empty())
    {
        //printf("source_image is not empty \n");
      source_image_resolution = source_image.size();
      Rect rect(0, 0, source_image_resolution.width, source_image_resolution.height);
      Subdiv2D subdiv(rect);
        //printf("line 446 passed \n");
        detectEyes(source_image);
        
        //printf("LEFT PUPIL: %f %f \n", leftPupil.x, leftPupil.y);
        //printf("RIGHT PUPIL: %f %f \n", rightPupil.x, rightPupil.y);
        draw_point(source_image, leftPupil, pupils_color);
        draw_point(source_image, rightPupil, pupils_color);


      face_landmark_element = face_landmark_list_head;
      while (face_landmark_element != NULL)
      {
          //printf("arguments for subdiv : %f %f \n", face_landmark_element->x, face_landmark_element->y);
          
          
          //printf("face_landmark_element index: %d \n", face_landmark_element->indice);
          
          if((face_landmark_element->indice > -1) && (face_landmark_element->indice < 68))
          {
              
              float correct_x = fabs(face_landmark_element->x);
              float correct_y = fabs(face_landmark_element->y);
              
              if(!isnan(correct_y) && !isnan(correct_x))
              {
                  if((correct_y > 0) && (correct_x > 0))
                  {
                      //printf("arguments for subdiv CORRECT_X CORRECT_Y : %f %f \n", correct_x, correct_y);
                      subdiv.insert(Point2f(correct_x, correct_y));
                  }
              }
          }
          else
          {
              break;
          }
          
          if(face_landmark_element->indice == 67)
          {
              break;
          }
          
          
          //printf("line 459 passed \n");
          if(face_landmark_element->next) //this should create the frame even without this
          {
        face_landmark_element = face_landmark_element->next;
          }
          else{
              break;
          }
      }
      draw_delaunay (source_image, subdiv, delaunay_color);
        //printf("line 463 passed \n");
      face_landmark_element = face_landmark_list_head;
      while (face_landmark_element != NULL)
      {
          //printf("arguments for draw_point : %f %f \n", face_landmark_element->x, face_landmark_element->y);
        draw_point (source_image, Point2f(face_landmark_element->x, face_landmark_element->y), points_color);
        
          if(face_landmark_element->next)
          {
          face_landmark_element = face_landmark_element->next;
          }
          else{
              break;
          }
      }
        //printf("about to print \n");
      imwrite (&output_filename[0], source_image);
    }
  }
}




face_landmark_node * add_face_landmark_element (face_landmark_node *face_landmark_list_head, int frame, int indice, float pixel_location_x, float pixel_location_y)
{
  face_landmark_node *new_face_landmark_element, *face_landmark_element, *previous_face_landmark_element;

  new_face_landmark_element = (face_landmark_node *) malloc (sizeof (face_landmark_node));
  if (new_face_landmark_element != NULL)
  {
      //printf("LINE 516 \n");
    new_face_landmark_element->frame = frame;
    new_face_landmark_element->indice = indice;
    new_face_landmark_element->x = pixel_location_x;
    new_face_landmark_element->y = pixel_location_y;
    new_face_landmark_element->next = NULL;
    if (face_landmark_list_head != NULL)
    {
        //printf("LINE 524 \n");
      face_landmark_element = face_landmark_list_head;
      while (face_landmark_element->next != NULL) 
      {
        face_landmark_element = face_landmark_element->next;
      }
        //printf("LINE 530 \n");
      face_landmark_element->next = new_face_landmark_element;

    }
    else
    {
      face_landmark_list_head = new_face_landmark_element;
    }
  }
    //printf("IT WENT THROGUH ADD_FACE_LANDMARK_ELEMENT \n");
  return face_landmark_list_head;
}



face_landmark_node * load_face_landmark_data (face_landmark_node *face_landmark_list_head)
{
  //face_landmark_list_head = add_face_landmark_element (face_landmark_list_head, 1, counter, x, y);
    int counter = 0;
    
    pixel_location_node *temp = (pixel_location_node *) malloc(sizeof(pixel_location_node));
    
    temp = pixel_head;
    
    while(temp != NULL)
    {
        //if it is eye corner find the mid
        
        //else
        //printf("temp pixel index: %d \n", temp->pixel_index);
        //printf("LOAD_FACE_LANDMARK DATA: %f %f \n", temp->pixel_location_x, temp->pixel_location_y);
        
        //printf("LINE 565 \n");
        
        if((temp->pixel_index > -1) && (temp->pixel_index < 68))
        {
            float correct_x = fabs(temp->pixel_location_x);
            float correct_y = fabs(temp->pixel_location_y);
            if(!isnan(correct_x) && !isnan(correct_y))
            {
                if((correct_y > 0) && (correct_x > 0))
                   {
                       //printf("LOAD_FACE_LANDMARK DATA CORRECT_X CORRECT_Y: %f %f \n", correct_x, correct_y);
            face_landmark_list_head = add_face_landmark_element (face_landmark_list_head, 1, counter, correct_x, correct_y);
                   }
            }
        }
        else
        {
            break;
        }
        
        if (temp->pixel_index == 67)
        {
            break;
        }

        //printf("ADDED_FACE_LANDMARK_ELEMENT \n");
        if(temp->next)
        {
            //printf("LINE 564 \n");
            temp = temp->next;
        }
        else
        {
            //printf("LINE 569 \n");
            break;
        }
        counter++;
        //printf("LINE 573 \n");
    }
    //printf("FINISHED LOAD_FACE_LANDARMARK_DATA \n");
  return face_landmark_list_head;
}


int parse_location(char *line) //returning 1 will stop scanning
{
    if(line_number < 4) //first open brace on line 3
    {
        return 0;
    }
    
    if(line_number > 71) //first close brace on line 72
    {
        return 1;
    }
    
    char *space_delim;
    const char *space = " ";
    
    space_delim = strtok(line, space);
    
    float px_x_y[2] = {0,0};
    
    //a counter for the linkedlist pixel_location's index
    //0 = didn't hit a curly brace 1 = did hit a curly brace
    int counter = 0;
    
    while(space_delim != NULL)
    {
        //printf("current space_delim: %s \n", space_delim);
        px_x_y[counter] = atof(space_delim);
        //printf("it went through the space_delim loop: %f \n", px_x_y[counter]);
        counter++;
        
        space_delim = strtok(NULL, space);
        
    }
    
    if(px_x_y[0] != 0.0 && px_x_y[1] != 0.0)
    {
        pixel_location_node *temp = (pixel_location_node *) malloc(sizeof(pixel_location_node));
        
        temp->pixel_index = pln_index;
        temp->pixel_location_x = px_x_y[0];
        temp->pixel_location_y = px_x_y[1];
        if(!pixel_head)
        {
            pixel_head = temp;
            pixel_current = temp;
        }
        else{
            pixel_current->next = temp;
            pixel_current = temp;
        }
        
        pln_index++;
    }

    return 0;
}


void load_points(char *file)
{
    FILE *fp;
    size_t bytes_read;
    char *file_buffer;
    int stop_scan = 0;
    
    fp = fopen (file, "r");
    if (fp != NULL)
    {
        //printf("FILE EXISTS IT'S LOADING POINTS \n");
        file_buffer = NULL;
        while (getline (&file_buffer, &bytes_read, fp) != -1)
        {
            //printf ("LOAD_POINTS FILE BUFFER STUFF: %s \n", file_buffer);
            stop_scan = parse_location(file_buffer);
            line_number++;
            //printf("line_number: %d \n", line_number);
            
            if(stop_scan == 1)
            {
                fclose (fp);
            }
            
            if (file_buffer != NULL)
            {
                free (file_buffer);
                file_buffer = NULL;
            }
        }
        fclose (fp);
    }
    
    //printf("END LOAD_POINTS \n");
}



int main (int argc, char *argv[]) 
{
    char *filename = argv[1]; //the image name
    load_points(argv[2]); //68 points file
    
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n"); return -1; };
    
  face_landmark_node *face_landmark_list_head, *face_landmark_element;

  face_landmark_list_head = NULL;
  face_landmark_list_head = load_face_landmark_data (face_landmark_list_head);
    //printf("LINE 668 \n");
  run (face_landmark_list_head, filename);
  while (face_landmark_list_head != NULL)
  {
    face_landmark_element = face_landmark_list_head;
    face_landmark_list_head = face_landmark_list_head->next;
    free (face_landmark_element);
    face_landmark_element = NULL;
  }
  exit (0);
}
