#import <UIKit/UIKit.h>
#import <opencv2/opencv.hpp>

@interface UIImage (CVMat)

-(cv::Mat) getCVMat;
-(cv::Mat) getCVRGBMat; //获取rgb mat
-(cv::Mat) getCVGrayscaleMat;
+ (UIImage *)initWithCVMat:(const cv::Mat&)cvMat;

@end
