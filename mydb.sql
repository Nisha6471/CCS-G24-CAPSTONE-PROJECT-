Create database biometric_data;
use biometric_data;
CREATE TABLE biometric_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    biometric_key BINARY(16),
    encrypted_data TEXT,
    iv TEXT
);
