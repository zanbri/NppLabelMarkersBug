////
////
//// Compile with: nvcc -o main main.cu -lnppif `pkg-config opencv --cflags --libs`
////
////


#include <cuda_runtime.h>
#include <npp.h>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

int main( int argc, char *argv[] )
{

    ///
    /// Set test filename
    ///
    const std::string inputFile = "test.png";

    //----------------------------------------------------------------------

    ///
    /// Read the input image on host
    ///
    cv::Mat inputImage = cv::imread( inputFile, CV_LOAD_IMAGE_GRAYSCALE ); // NOTE type of image to read

    assert( inputImage.data && "Could not open or find input image file" );

    const int mW = inputImage.cols;
    const int mH = inputImage.rows;
    printf( "The size of the input image: (%d,%d)\n", mW, mH );

    //----------------------------------------------------------------------

    ///
    /// Allocate memory on device
    ///
    Npp8u  *d_iImage;
    Npp32u *d_tImage;
    cudaMalloc( (void **)(&d_iImage), mW*mH*sizeof(Npp8u)  );
    cudaMalloc( (void **)(&d_tImage), mW*mH*sizeof(Npp32u) );

    //----------------------------------------------------------------------

    ///
    /// Transfer data to device
    ///
    cudaMemcpy( d_iImage, inputImage.data, mW*mH*sizeof(Npp8u), cudaMemcpyHostToDevice );

    //----------------------------------------------------------------------

    int max;
    NppiSize srcSize = { mW, mH };
    NppStatus npp_err;

    ///
    /// Get buffer size
    ///
    int nBufferSize = 0;
    npp_err = nppiLabelMarkersGetBufferSize_8u32u_C1R( srcSize, &nBufferSize );
    assert( npp_err == NPP_SUCCESS );

    // Allocate the scratch buffer 
    Npp8u *pBuffer = 0;
    cudaMalloc( (void **)(&pBuffer), nBufferSize );

    printf( "Buffer size 1: %d\n", nBufferSize );

    ///
    /// Connected components labelling
    ///
    npp_err = nppiLabelMarkers_8u32u_C1R(
            d_iImage,
            mW*sizeof(Npp8u),
            d_tImage,
            mW*sizeof(Npp32u),
            srcSize,
            (Npp8u)0,
            nppiNormInf, // 8-way connectivity
            &max,
            pBuffer
            );
    assert( npp_err == NPP_SUCCESS );

    printf( "Max 1: %d\n", max );

    //-----------------------------------------------------------------------------------------

    ///
    /// Get buffer size
    ///
    int nCompressBufferSize = 0;
    npp_err = nppiCompressMarkerLabelsGetBufferSize_32u8u_C1R( max, &nCompressBufferSize );
    assert( npp_err == NPP_SUCCESS );
    
    if( nCompressBufferSize > nBufferSize ) {
        nBufferSize = nCompressBufferSize;
        cudaFree( pBuffer );
        cudaMalloc( &pBuffer, nBufferSize );
    }

    printf( "Buffer size 2: %d\n", nBufferSize );

    ///
    /// Compress marker labels
    ///
    npp_err = nppiCompressMarkerLabels_32u8u_C1R(
            d_tImage,
            mW*sizeof(Npp32u),
            d_iImage,
            mW*sizeof(Npp8u),
            srcSize,
            max,
            &max,
            pBuffer
            );
    
    assert( npp_err == NPP_SUCCESS );

    printf( "Max 2: %d\n", max );

    assert( max < 256 && "Number of connected components found exceeds limit." );
    
    // Transfer output to host
    cv::Mat h_oImage( mH, mW, 0 );
    cudaMemcpy( h_oImage.data, d_iImage, mW*mH*sizeof(Npp8u), cudaMemcpyDeviceToHost );

    cv::imwrite( "output.png", h_oImage );

    //----------------------------------------------------------------------

    ///
    /// Clean up
    ///
    cudaFree( d_iImage );
    cudaFree( d_tImage );
    cudaFree( pBuffer );

    return 0;

}
