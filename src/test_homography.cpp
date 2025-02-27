#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

auto main() -> int {
  // 1. Load images in color
  cv::Mat img1 = cv::imread("image1.jpeg", cv::IMREAD_COLOR); // Template image
  cv::Mat img2 = cv::imread("image2.jpeg", cv::IMREAD_COLOR); // Scene image
  if(img1.empty() || img2.empty()) {
    std::cerr << "Error loading images.\n";
    return -1;
  }
  std::cout << "Images loaded.\n";

  // Convert to grayscale for feature detection
  cv::Mat img1Gray, img2Gray;
  cv::cvtColor(img1, img1Gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(img2, img2Gray, cv::COLOR_BGR2GRAY);

  // 2. Detect keypoints and compute descriptors using ORB
  cv::Ptr< cv::ORB > detector = cv::ORB::create();
  std::vector< cv::KeyPoint > kp1, kp2;
  cv::Mat desc1, desc2;
  detector->detectAndCompute(img1Gray, cv::noArray(), kp1, desc1);
  detector->detectAndCompute(img2Gray, cv::noArray(), kp2, desc2);
  std::cout << "Keypoints detected: img1=" << kp1.size()
            << ", img2=" << kp2.size() << "\n";

  // 3. Match descriptors using BFMatcher with cross-check
  cv::BFMatcher matcher(cv::NORM_HAMMING, true);
  std::vector< cv::DMatch > matches;
  matcher.match(desc1, desc2, matches);
  std::cout << "Matches found: " << matches.size() << "\n";

  // 4. Filter good matches based on distance
  double max_dist = 0;
  for(const auto& m : matches) {
    if(m.distance > max_dist) {
      max_dist = m.distance;
    }
  }
  std::vector< cv::DMatch > goodMatches;
  for(const auto& m : matches) {
    if(m.distance < 0.6 * max_dist) {
      goodMatches.push_back(m);
    }
  }
  std::cout << "Good matches filtered: " << goodMatches.size() << "\n";

  // 5. Extract location of good matches
  std::vector< cv::Point2f > pts1, pts2;
  for(const auto& m : goodMatches) {
    pts1.push_back(kp1[m.queryIdx].pt);
    pts2.push_back(kp2[m.trainIdx].pt);
  }

  // 6. Find homography
  cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC);
  if(H.empty()) {
    std::cerr << "Homography computation failed.\n";
    return -1;
  }
  std::cout << "Homography matrix:\n" << H << "\n";

  // 7. Decompose homography to get rotation and translation
  cv::Mat K = cv::Mat::eye(
      3, 3, CV_64F); // Assuming camera intrinsic matrix is identity
  std::vector< cv::Mat > rotations, translations, normals;
  int solutions
      = cv::decomposeHomographyMat(H, K, rotations, translations, normals);
  std::cout << "Number of solutions: " << solutions << "\n";

  // 8. Select the first solution (for simplicity)
  cv::Mat R = rotations[0];
  cv::Mat t = translations[0];
  std::cout << "Rotation matrix:\n" << R << "\n";
  std::cout << "Translation vector:\n" << t << "\n";

  // 9. Draw bounding box around detected object in both images
  std::vector< cv::Point2f > objCorners(4);
  objCorners[0] = cv::Point2f(0, 0);
  objCorners[1] = cv::Point2f(static_cast< float >(img1.cols), 0);
  objCorners[2] = cv::Point2f(static_cast< float >(img1.cols),
                              static_cast< float >(img1.rows));
  objCorners[3] = cv::Point2f(0, static_cast< float >(img1.rows));

  // Draw box in img1 (template)
  cv::line(img1, objCorners[0], objCorners[1], cv::Scalar(0, 255, 0), 4);
  cv::line(img1, objCorners[1], objCorners[2], cv::Scalar(0, 255, 0), 4);
  cv::line(img1, objCorners[2], objCorners[3], cv::Scalar(0, 255, 0), 4);
  cv::line(img1, objCorners[3], objCorners[0], cv::Scalar(0, 255, 0), 4);

  // Draw transformed box in img2 (scene)
  std::vector< cv::Point2f > sceneCorners(4);
  cv::perspectiveTransform(objCorners, sceneCorners, H);
  cv::line(img2, sceneCorners[0], sceneCorners[1], cv::Scalar(0, 255, 0), 4);
  cv::line(img2, sceneCorners[1], sceneCorners[2], cv::Scalar(0, 255, 0), 4);
  cv::line(img2, sceneCorners[2], sceneCorners[3], cv::Scalar(0, 255, 0), 4);
  cv::line(img2, sceneCorners[3], sceneCorners[0], cv::Scalar(0, 255, 0), 4);

  // 10. Display images with matches and bounding boxes
  cv::Mat imgMatches;
  cv::drawMatches(img1,
                  kp1,
                  img2,
                  kp2,
                  goodMatches,
                  imgMatches,
                  cv::Scalar::all(-1),
                  cv::Scalar::all(-1),
                  std::vector< char >(),
                  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  cv::imshow("Good Matches & Object Detection", imgMatches);
  cv::waitKey(0);

  return 0;
}
