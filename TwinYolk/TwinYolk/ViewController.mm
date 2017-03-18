//
//  ViewController.m
//  TwinYolk
//
//  Created by darrenyao on 2017/3/18.
//  Copyright © 2017年 tencent. All rights reserved.
//

#import "ViewController.h"
#import "UIImage+CVMat.h"
#import <opencv2/opencv.hpp>
#import <opencv2/nonfree/nonfree.hpp>

using namespace cv;

@interface ViewController ()
@property (nonatomic, weak) IBOutlet UIImageView *imageViewL;
@property (nonatomic, weak) IBOutlet UIImageView *imageViewR;
@property (nonatomic, weak) IBOutlet UIImageView *imageViewBlend;


@property (nonatomic, strong) UIImage *imageL;
@property (nonatomic, strong) UIImage *imageR;
@property (nonatomic, strong) UIImage *imageBlend;
@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    
    NSString *imagePathL = [[NSBundle mainBundle] pathForResource:@"test_1.jpg"
                                                           ofType:nil];
    NSString *imagePathR = [[NSBundle mainBundle] pathForResource:@"test_2.jpg"
                                                           ofType:nil];
    
    _imageL = [UIImage imageWithContentsOfFile:imagePathL];
    _imageR = [UIImage imageWithContentsOfFile:imagePathR];
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [self testImageBlend];
        dispatch_async(dispatch_get_main_queue(), ^{
            self.imageViewL.image = self.imageL;
            self.imageViewR.image = self.imageR;
            self.imageViewBlend.image = self.imageBlend;
        });
    });
}

- (void)testImageBlend {
    cv::Mat matBlend;
    cv::Mat matL = _imageL.getCVMat;
    cv::Mat matR = _imageR.getCVMat;
//    cv::cvtColor(matL, matL, CV_RGBA2RGB);
//    cv::cvtColor(matR, matR, CV_RGBA2RGB);
    
    cv::Mat grayL, grayR;
    cv::cvtColor(matL, grayL, CV_RGBA2GRAY);
    cv::cvtColor(matR, grayR, CV_RGBA2GRAY);
    
    std::vector<cv::Mat> imgs;
    imgs.push_back(matL);
    imgs.push_back(matR);
    
    int minHessian = 400;
    SurfFeatureDetector detector(minHessian);
    vector<KeyPoint> keyPointsL, keyPointsR;
    
    detector.detect(grayL, keyPointsL);
    detector.detect(grayR, keyPointsR);
    
    SurfDescriptorExtractor descripExtractor;
    cv::Mat descripsL, descripsR;
    descripExtractor.compute(grayL, keyPointsL, descripsL);
    descripExtractor.compute(grayR, keyPointsR, descripsR);
    
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match(descripsL, descripsR, matches);
    size_t matchCount = matches.size();
    sort(matches.begin(),matches.begin()+matchCount);
    
    vector<Point2f> pointsL;
    vector<Point2f> pointsR;
    
    for(int i=0; i<matchCount; i++) {
        
        pointsL.push_back(keyPointsL[matches[i].queryIdx].pt);
        pointsR.push_back(keyPointsR[matches[i].trainIdx].pt);
        
    }
    
    cv::Mat homo = cv::findHomography(pointsL, pointsR, RANSAC);
    cv::Mat shftMat = (cv::Mat_<double>(3,3)<<1.0,0,grayL.cols, 0,1.0,0, 0,0,1.0);
    
    
    
    //拼接图像
    
    Mat matTitled;
    
    cv::warpPerspective(matL, matTitled, homo, cv::Size(matL.cols+matR.cols, matR.rows));
    matR.copyTo(Mat(matTitled, cv::Rect(matL.cols, 0, matR.cols, matR.rows)));
    
    self.imageBlend = [UIImage initWithCVMat:matTitled];
//    Ptr<FeatureDetector> detector = FeatureDetector::create("SIFT");
//    Ptr<DescriptorExtractor> descpExtractor = DescriptorExtractor::create("SIFT");
//    Ptr<DescriptorMatcher> descpMatcher = DescriptorMatcher::create("BruteForce");
//    S
//    
//    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::Mode::SCANS, false);
//    Stitcher::Status status = stitcher->stitch(imgs, matBlend);
    
//    seamlessClone(<#InputArray src#>, <#InputArray dst#>, <#InputArray mask#>, <#Point p#>, <#OutputArray blend#>, <#int flags#>)
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
