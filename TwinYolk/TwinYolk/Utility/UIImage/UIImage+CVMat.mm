#import "UIImage+CVMat.h"

@implementation UIImage (CVMat)

-(cv::Mat) getCVMat
{
    if (self == nil || self.CGImage == nil) {
        return cv::Mat();
    }
    CGColorSpaceRef colorSpace = CGImageGetColorSpace(self.CGImage);
    int cols = (int)CGImageGetWidth(self.CGImage);
    int rows = (int)CGImageGetHeight(self.CGImage);
    
    if (cols == 0 || rows == 0) {
        return cv::Mat();
    }
    
    cv::Mat cvMat;
    cvMat.create(rows, cols, CV_8UC4); // 8 bits per component, 4 channels
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to backing data
                                                    cols,                      // Width of bitmap
                                                    rows,                     // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNoneSkipLast |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), self.CGImage);
    CGContextRelease(contextRef);
    
    return cvMat;
}
-(cv::Mat) getCVRGBMat
{
    if (self == nil || self.CGImage == nil) {
        return cv::Mat();
    }
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    int cols = (int)CGImageGetWidth(self.CGImage);
    int rows = (int)CGImageGetHeight(self.CGImage);
    
    if (cols == 0 || rows == 0) {
        CGColorSpaceRelease(colorSpace);
        return cv::Mat();
    }
    
    cv::Mat cvMat;
    cvMat.create(rows, cols, CV_8UC3); // 8 bits per component, 4 channels
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to backing data
                                                    cols,                      // Width of bitmap
                                                    rows,                     // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNone |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), self.CGImage);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    return cvMat;
}

-(cv::Mat) getCVGrayscaleMat
{
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceGray();
    CGFloat cols = (int)CGImageGetWidth(self.CGImage);
    CGFloat rows = (int)CGImageGetHeight(self.CGImage);
    
    cv::Mat cvMat = cv::Mat(rows, cols, CV_8UC1); // 8 bits per component, 1 channel
    
    CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to backing data
                                                    cols,                      // Width of bitmap
                                                    rows,                     // Height of bitmap
                                                    8,                          // Bits per component
                                                    cvMat.step[0],              // Bytes per row
                                                    colorSpace,                 // Colorspace
                                                    kCGImageAlphaNone |
                                                    kCGBitmapByteOrderDefault); // Bitmap info flags
    
    CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), self.CGImage);
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    return cvMat;
}

+ (UIImage *)initWithCVMat:(const cv::Mat&)cvMat
{
    
    cv::Mat rgbaMat = cvMat;
    if (3 == cvMat.channels()) {
        cv::cvtColor(cvMat, rgbaMat, CV_RGB2RGBA);
    }
    
    NSData *data = [NSData dataWithBytes:rgbaMat.data length:rgbaMat.elemSize() * rgbaMat.total()];
    
    CGColorSpaceRef colorSpace;
    
    if (rgbaMat.elemSize() == 1)
    {
        colorSpace = CGColorSpaceCreateDeviceGray();
    }
    else
    {
        colorSpace = CGColorSpaceCreateDeviceRGB();
    }
    
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data);
    
    CGImageRef imageRef = CGImageCreate(rgbaMat.cols,                                     // Width
                                        rgbaMat.rows,                                     // Height
                                        8,                                              // Bits per component
                                        8 * rgbaMat.elemSize(),                           // Bits per pixel
                                        rgbaMat.step[0],                                  // Bytes per row
                                        colorSpace,                                     // Colorspace
                                        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault,  // Bitmap info flags
                                        provider,                                       // CGDataProviderRef
                                        NULL,                                           // Decode
                                        false,                                          // Should interpolate
                                        kCGRenderingIntentDefault);                     // Intent
    
    UIImage *uiImage = [UIImage imageWithCGImage:imageRef];
    
    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    
    return uiImage;
}

@end
