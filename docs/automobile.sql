
CREATE DATABASE automobile_india;
USE automobile_india;

-- Users table
CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(15),
    password VARCHAR(255),
    role ENUM('admin', 'staff', 'customer') DEFAULT 'customer',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Features table
CREATE TABLE features (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    icon VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Services table
CREATE TABLE services (
    id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    service_number INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Brands table
CREATE TABLE brands (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    model VARCHAR(100),
    price DECIMAL(12,2),
    image_url VARCHAR(500),
    specs JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Staff table
CREATE TABLE staff (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    role VARCHAR(100),
    phone VARCHAR(15),
    image_url VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Testimonials table
CREATE TABLE testimonials (
    id INT AUTO_INCREMENT PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    rating INT DEFAULT 5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Bookings table
CREATE TABLE bookings (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    service_type VARCHAR(100),
    car_model VARCHAR(100),
    booking_date DATE,
    status ENUM('pending', 'confirmed', 'completed') DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Insert sample data
INSERT INTO features (title, description, icon) VALUES
('Wide Range of Vehicles', 'From premium brands like BMW, Mercedes-Benz, and Audi to budget-friendly options like Tata, Hyundai, and Maruti Suzuki.', 'fas fa-car'),
('Verified & Certified', 'We connect you only with verified dealers and certified sellers, ensuring a safe, secure, and hassle-free buying or selling experience.', 'fas fa-shield-alt'),
('Best Price Guarantee', 'With our best price guarantees, you get access to exclusive offers, discounts, and wholesale deals on both new and used cars.', 'fas fa-tag');

INSERT INTO services (title, description, service_number) VALUES
('Online Vehicle Rental Booking', 'Easily book vehicles online with our user-friendly platform. Choose from a wide range of cars and bikes at the best prices.', 1),
('Professional Technician Services', 'Our certified technicians provide reliable inspection and maintenance services. We ensure quick response times and professional solutions.', 2),
('Full Auto Servicing Facility', 'Comprehensive auto care under one roof — from routine maintenance to major repairs. Our workshop is equipped with modern diagnostic tools.', 3),
('Pick And Drop Facility', 'We provide a convenient pick and drop facility for your vehicle servicing. Our team ensures timely pickup and hassle-free delivery.', 4);

INSERT INTO brands (name, model, price, specs) VALUES
('JAGUAR', 'F-PACE', 7800000, '{"starts": "24", "ac": "4", "mileage": "6.99km"}'),
('BMW', 'X1', 5590000, '{"starts": "24", "ac": "4", "mileage": "6.99km"}'),
('HONDA', 'City', 1590000, '{"starts": "24", "ac": "4", "mileage": "7.10km"}');

INSERT INTO staff (name, role, phone) VALUES
('Michael William', 'Business Agent', '8792024754'),
('Riya', 'Sales Executive', '758449029'),
('Jhon Palu', 'Automobile Technician', '907865521'),
('Lisa', 'HR Manager', '679007345');

INSERT INTO testimonials (customer_name, content, rating) VALUES
('Sofia', 'Professional technicians who really know their job. My car feels brand new after every service. Excellent!', 5),
('Joy', 'Found my dream car at an amazing price. The team was helpful throughout the buying process. Highly recommended!', 5),
('Harden', 'The pick and drop service saved me so much time. Professional staff and quality service. Will definitely come back.', 5);
-- Create database

-- Payments table
CREATE TABLE payments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    transaction_id VARCHAR(100) UNIQUE NOT NULL,
    car_model VARCHAR(255) NOT NULL,
    car_price DECIMAL(12, 2) NOT NULL,
    base_price DECIMAL(12, 2) NOT NULL,
    taxes DECIMAL(12, 2) NOT NULL,
    total_amount DECIMAL(12, 2) NOT NULL,
    currency VARCHAR(10) DEFAULT 'INR',
    
    -- Customer information (stored as JSON or separate columns)
    customer_name VARCHAR(255) NOT NULL,
    customer_email VARCHAR(255) NOT NULL,
    customer_phone VARCHAR(20),
    customer_address TEXT,
    
    -- Payment details
    payment_method ENUM('credit_card', 'debit_card', 'upi', 'net_banking') NOT NULL,
    card_last_four VARCHAR(4),
    card_type VARCHAR(50),
    
    -- Status and timestamps
    status ENUM('pending', 'completed', 'failed', 'refunded') DEFAULT 'pending',
    payment_gateway VARCHAR(100) DEFAULT 'AutoPay Gateway',
    gateway_transaction_id VARCHAR(100),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_transaction_id (transaction_id),
    INDEX idx_customer_email (customer_email),
    INDEX idx_created_at (created_at),
    INDEX idx_status (status)
);

-- Transactions log table for audit
CREATE TABLE transaction_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    payment_id INT,
    transaction_id VARCHAR(100) NOT NULL,
    log_type ENUM('payment_initiated', 'payment_processed', 'payment_failed', 'refund_initiated'),
    log_message TEXT,
    log_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (payment_id) REFERENCES payments(id) ON DELETE CASCADE,
    INDEX idx_transaction_id (transaction_id)
);

-- Refunds table
CREATE TABLE refunds (
    id INT AUTO_INCREMENT PRIMARY KEY,
    payment_id INT,
    refund_amount DECIMAL(12, 2) NOT NULL,
    refund_reason TEXT,
    status ENUM('pending', 'processed', 'failed') DEFAULT 'pending',
    refund_transaction_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (payment_id) REFERENCES payments(id) ON DELETE CASCADE
);