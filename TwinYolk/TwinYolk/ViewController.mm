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
    cv::cvtColor(matL, matL, CV_RGBA2RGB);
    cv::cvtColor(matR, matR, CV_RGBA2RGB);
    
//    cv::Mat grayL, grayR;
//    cv::cvtColor(rgbL, grayL, CV_RGBA2GRAY);
//    cv::cvtColor(rgbR, grayR, CV_RGBA2GRAY);
    
    std::vector<cv::Mat> imgs;
    imgs.push_back(matL);
    imgs.push_back(matR);
    
    S
    
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::Mode::SCANS, false);
    Stitcher::Status status = stitcher->stitch(imgs, matBlend);
    
//    seamlessClone(<#InputArray src#>, <#InputArray dst#>, <#InputArray mask#>, <#Point p#>, <#OutputArray blend#>, <#int flags#>)
}


- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


@end
