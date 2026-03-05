#ifndef NETWORK_UTILS_HPP
#define NETWORK_UTILS_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>

inline int establish_connection(const std::string& host, int port) {
    struct addrinfo hints{}, *res;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    std::string port_str = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_str.c_str(), &hints, &res) != 0) return -1;
    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd >= 0) {
        if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) { close(fd); fd = -1; }
    }
    freeaddrinfo(res);
    return fd;
}

inline std::string simple_http_post_persistent(int& sockfd, const std::string& host, int port, const std::string& path, const std::string& body, const std::string& content_type) {
    if (sockfd < 0) sockfd = establish_connection(host, port);
    if (sockfd < 0) return "";

    std::ostringstream req;
    req << "POST " << path << " HTTP/1.1\r\n"
        << "Host: " << host << "\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: keep-alive\r\n\r\n" << body;
    std::string request_str = req.str();

    // std::cerr << "\n--- [START RAW REQUEST] ---\n" << request_str << "\n--- [END RAW REQUEST] ---\n" << std::endl;

    if (send(sockfd, request_str.c_str(), request_str.size(), MSG_NOSIGNAL) < 0) {
        close(sockfd);
        sockfd = establish_connection(host, port);
        if (sockfd < 0 || send(sockfd, request_str.c_str(), request_str.size(), MSG_NOSIGNAL) < 0) return "";
    }

    std::string response;
    char buffer[4096];
    size_t header_end = std::string::npos;
    long content_length = -1;

    while (header_end == std::string::npos) {
        ssize_t n = recv(sockfd, buffer, sizeof(buffer), 0);
        if (n <= 0) { close(sockfd); sockfd = -1; return ""; }
        response.append(buffer, n);
        header_end = response.find("\r\n\r\n");
    }

    size_t cl_pos = response.find("Content-Length:");
    if (cl_pos == std::string::npos) cl_pos = response.find("content-length:");
    if (cl_pos != std::string::npos) {
        size_t start = response.find_first_of("0123456789", cl_pos);
        size_t end = response.find("\r\n", start);
        content_length = std::stol(response.substr(start, end - start));
    }

    size_t body_start = header_end + 4;
    while (content_length > 0 && (response.size() < body_start + content_length)) {
        ssize_t n = recv(sockfd, buffer, sizeof(buffer), 0);
        if (n <= 0) break;
        response.append(buffer, n);
    }

    // std::cerr << "\n--- [START RAW RESPONSE] ---\n" << response << "\n--- [END RAW RESPONSE] ---\n" << std::endl;
    return response.substr(body_start, content_length);
}

#endif
