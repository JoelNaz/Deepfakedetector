import os
import time
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.parsers import MultiPartParser, FileUploadParser
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
from .models import VideoAnalysis
from .utils import get_detector
from django.core.paginator import Paginator
from datetime import datetime, timedelta



@api_view(['POST'])
@permission_classes([IsAuthenticated])
@parser_classes([MultiPartParser, FileUploadParser])
def analyze_video(request):
    try:
        # Check if video file is provided
        if 'video' not in request.FILES:
            return Response({
                'status': 'error',
                'message': 'No video file provided'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        video_file = request.FILES['video']
        
        # Validate file type
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        file_extension = os.path.splitext(video_file.name)[1].lower()
        
        if file_extension not in allowed_extensions:
            return Response({
                'status': 'error',
                'message': f'Unsupported file format. Allowed formats: {", ".join(allowed_extensions)}'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Check file size (limit to 100MB)
        max_size = 100 * 1024 * 1024  # 100MB
        if video_file.size > max_size:
            return Response({
                'status': 'error',
                'message': 'File size too large. Maximum allowed size is 100MB'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Save the uploaded file temporarily
        timestamp = str(int(time.time()))
        filename = f"{timestamp}_{video_file.name}"
        file_path = default_storage.save(f'videos/{filename}', ContentFile(video_file.read()))
        full_file_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
        try:
            # Get detector instance and analyze video
            detector = get_detector()
            results = detector.analyze_video(full_file_path)
            
            # Save analysis results to database
            analysis = VideoAnalysis.objects.create(
                user=request.user,
                video_file=file_path,
                filename=video_file.name,
                file_size=video_file.size,
                prediction_score=results['prediction_score'],
                is_deepfake=results['is_deepfake'],
                confidence_threshold=results['threshold'],
                analysis_duration=results['analysis_duration'],
                frames_analyzed=results['frames_per_video']
            )
            
            # Prepare response
            response_data = {
                'status': 'success',
                'message': 'Video analysis completed',
                'analysis_id': analysis.id,
                'filename': video_file.name,
                'is_fake': results['is_deepfake'],
                'confidence': results['confidence'],
                'raw_score': results['prediction_score'],
                'threshold': results['threshold'],
                'analysis_duration': results['analysis_duration']
            }
            
            return Response(response_data, status=status.HTTP_200_OK)
            
        except Exception as e:
            # Clean up the uploaded file if analysis fails
            if os.path.exists(full_file_path):
                os.remove(full_file_path)
            
            return Response({
                'status': 'error',
                'message': f'Analysis failed: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        finally:
            # Clean up the temporary file after analysis
            if os.path.exists(full_file_path):
                try:
                    os.remove(full_file_path)
                except:
                    pass  # Ignore cleanup errors
    
    except Exception as e:
        return Response({
            'status': 'error',
            'message': f'Unexpected error: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def analysis_history(request):
    try:
        # Get user's analysis history
        analyses = VideoAnalysis.objects.filter(user=request.user).order_by('-created_at')[:20]
        
        history_data = []
        for analysis in analyses:
            history_data.append({
                'id': analysis.id,
                'filename': analysis.filename,
                'is_deepfake': analysis.is_deepfake,
                'confidence_percentage': analysis.confidence_percentage,
                'prediction_score': analysis.prediction_score,
                'created_at': analysis.created_at.isoformat(),
                'analysis_duration': analysis.analysis_duration
            })
        
        return Response({
            'status': 'success',
            'history': history_data
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        return Response({
            'status': 'error',
            'message': f'Failed to fetch history: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def analysis_detail(request, analysis_id):
    try:
        # Get specific analysis details
        analysis = VideoAnalysis.objects.get(id=analysis_id, user=request.user)
        
        analysis_data = {
            'id': analysis.id,
            'filename': analysis.filename,
            'file_size': analysis.file_size,
            'is_deepfake': analysis.is_deepfake,
            'confidence_percentage': analysis.confidence_percentage,
            'prediction_score': analysis.prediction_score,
            'confidence_threshold': analysis.confidence_threshold,
            'frames_analyzed': analysis.frames_analyzed,
            'analysis_duration': analysis.analysis_duration,
            'created_at': analysis.created_at.isoformat(),
            'updated_at': analysis.updated_at.isoformat()
        }
        
        return Response({
            'status': 'success',
            'analysis': analysis_data
        }, status=status.HTTP_200_OK)
        
    except VideoAnalysis.DoesNotExist:
        return Response({
            'status': 'error',
            'message': 'Analysis not found'
        }, status=status.HTTP_404_NOT_FOUND)
        
    except Exception as e:
        return Response({
            'status': 'error',
            'message': f'Failed to fetch analysis details: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def analysis_history(request):
    """Get user's analysis history with pagination and filtering"""
    try:
        # Get user's analyses ordered by newest first
        analyses = VideoAnalysis.objects.filter(user=request.user).order_by('-created_at')
        
        # Optional filtering
        result_filter = request.GET.get('result')  # 'real', 'fake', or 'all'
        if result_filter and result_filter != 'all':
            if result_filter == 'real':
                analyses = analyses.filter(is_deepfake=False)
            elif result_filter == 'fake':
                analyses = analyses.filter(is_deepfake=True)
        
        # Optional date filtering
        days = request.GET.get('days')  # last 7, 30, 90 days
        if days:
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=int(days))
            analyses = analyses.filter(created_at__gte=cutoff_date)
        
        # Pagination
        page = int(request.GET.get('page', 1))
        per_page = int(request.GET.get('per_page', 10))
        paginator = Paginator(analyses, per_page)
        
        try:
            analyses_page = paginator.page(page)
        except:
            analyses_page = paginator.page(1)
        
        # Format response
        history_data = []
        for analysis in analyses_page:
            history_data.append({
            'id': analysis.id,
            'filename': analysis.filename,
            'created_at': analysis.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'date_only': analysis.created_at.strftime('%Y-%m-%d'),
            'time_only': analysis.created_at.strftime('%H:%M'),
            'is_deepfake': analysis.is_deepfake,
            'result_text': 'Deepfake' if analysis.is_deepfake else 'Real',
            'confidence': round(analysis.confidence_percentage, 1),  # ← FIXED
            'prediction_score': round(analysis.prediction_score, 3),
            'analysis_duration': round(analysis.analysis_duration, 1),
            'frames_analyzed': analysis.frames_analyzed or 16
        })
        
        return Response({
            'success': True,
            'history': history_data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total_pages': paginator.num_pages,
                'total_count': paginator.count,
                'has_next': analyses_page.has_next(),
                'has_previous': analyses_page.has_previous()
            },
            'stats': {
                'total_analyses': VideoAnalysis.objects.filter(user=request.user).count(),
                'real_count': VideoAnalysis.objects.filter(user=request.user, is_deepfake=False).count(),
                'fake_count': VideoAnalysis.objects.filter(user=request.user, is_deepfake=True).count(),
            }
        })
        
    except Exception as e:
        return Response({
            'success': False,
            'error': str(e)
        }, status=500)