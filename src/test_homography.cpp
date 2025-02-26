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
    if(dist < min_dist)
      min_dist = dist;
    if(dist > max_dist)
      max_dist = dist;
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

  // Create base image for matches
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

  int current_match = 0;
  cv::namedWindow("Matches", cv::WINDOW_NORMAL);

  while(true) {
    // Create a copy of the base image for drawing current match
    cv::Mat current_display = img_matches.clone();

    if(current_match < good_matches.size()) {
      cv::Point2f point1 = keypoints1[good_matches[current_match].queryIdx].pt;
      cv::Point2f point2 = keypoints2[good_matches[current_match].trainIdx].pt;
      point2.x += static_cast< float >(img1.cols);

      // Use a fixed bright color for better visibility
      cv::Scalar color(0, 255, 255); // Yellow color

      // Draw much thicker line for selected match
      cv::line(current_display, point1, point2, color, 8);

      // Draw larger circles at keypoints
      cv::circle(current_display, point1, 15, color, -1);
      cv::circle(current_display, point2, 15, color, -1);

      // Display match number
      std::string match_text = "Match " + std::to_string(current_match + 1)
                               + " of " + std::to_string(good_matches.size());
      cv::putText(current_display,
                  match_text,
                  cv::Point(20, 40),
                  cv::FONT_HERSHEY_SIMPLEX,
                  1.0,
                  cv::Scalar(0, 255, 0),
                  2);
    }

    cv::imshow("Matches", current_display);

    // Wait for keypress
    char key = cv::waitKey(0);
    if(key == 27) { // ESC key to exit
      break;
    } else if(key == 'n' || key == 'N') { // Next match
      current_match = (current_match + 1) % good_matches.size();
    } else if(key == 'b' || key == 'B') { // Previous match
      current_match
          = (current_match - 1 + good_matches.size()) % good_matches.size();
    }
  }

  return 0;
}
