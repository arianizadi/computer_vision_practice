#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

auto main() -> int {
  // Load the images
  cv::Mat img1 = cv::imread("image1.jpeg");
  cv::Mat img2 = cv::imread("image2.jpeg");

  if(img1.empty() || img2.empty()) {
    std::cout << "Error: Could not load one or both images" << std::endl;
    return -1;
  }

  // Initialize ORB detector
  cv::Ptr< cv::ORB > orb = cv::ORB::create();

  // Detect keypoints and compute descriptors
  std::vector< cv::KeyPoint > keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;

  orb->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
  orb->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

  // Match features using BFMatcher
  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector< cv::DMatch > matches;
  matcher.match(descriptors1, descriptors2, matches);

  // Find good matches using distance threshold
  double max_dist = 0;
  double min_dist = 100;

  for(int i = 0; i < descriptors1.rows; i++) {
    double dist = matches[i].distance;
    if(dist < min_dist) {
      min_dist = dist;
    }
    if(dist > max_dist) {
      max_dist = dist;
    }
  }

  std::vector< cv::DMatch > good_matches;
  for(int i = 0; i < descriptors1.rows; i++) {
    if(matches[i].distance <= std::max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }

  // Extract location of good matches
  std::vector< cv::Point2f > points1, points2;
  for(size_t i = 0; i < good_matches.size(); i++) {
    points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
    points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
  }

  // Find homography matrix
  cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC);

  // Draw matches
  cv::Mat img_matches;
  cv::drawMatches(img1,
                  keypoints1,
                  img2,
                  keypoints2,
                  good_matches,
                  img_matches,
                  cv::Scalar::all(-1),
                  cv::Scalar::all(-1),
                  std::vector< char >(),
                  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  // Draw lines between matched points
  for(auto &good_matche : good_matches) {
    cv::Point2f point1 = keypoints1[good_matche.queryIdx].pt;
    cv::Point2f point2 = keypoints2[good_matche.trainIdx].pt;
    point2.x
        += static_cast< float >(img1.cols); // Adjust for side-by-side display
    cv::line(img_matches, point1, point2, cv::Scalar(0, 255, 0), 1);
  }

  // Show results
  cv::imshow("Matches", img_matches);
  cv::waitKey(0);

  return 0;
}
