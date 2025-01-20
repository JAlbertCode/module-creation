import unittest
import sys
import os
import logging
from datetime import datetime

def setup_logging():
    """Set up logging configuration"""
    log_dir = 'test_logs'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'test_run_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger('test_runner')

def run_tests():
    """Run all tests and generate report"""
    logger = setup_logging()
    
    # Start test session
    logger.info("Starting test session")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir)
    
    # Run tests and capture results
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Log results
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Skipped: {len(result.skipped)}")
    
    # Log failures in detail
    if result.failures:
        logger.error("Test Failures:")
        for failure in result.failures:
            logger.error(f"\n{failure[0]}\n{failure[1]}")
    
    # Log errors in detail
    if result.errors:
        logger.error("Test Errors:")
        for error in result.errors:
            logger.error(f"\n{error[0]}\n{error[1]}")
    
    return len(result.failures) + len(result.errors)

if __name__ == '__main__':
    sys.exit(run_tests())